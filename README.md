# garage_recon：多层地下车库 车载 3D 重建 + 自动语义标注

对应方案中的 **第四部分（车载重建）** 和 **第五部分（自动语义标注）**。
自带一个合成车库数据生成器（两层 + 坡道 + 停放/移动车辆 + 车载 LiDAR/鱼眼/IMU/轮速 + 真值），整条链路不依赖真实数据也能跑通和量化评测。

> 方案 v2（2026-10-12）见 [docs/plan.md](docs/plan.md)：几何改为 LiDAR 骨架 + DA3 + 3DGS，语义改为 SAM + SigLIP / CLIP + ILGS 式语言高斯，跨视角对应用 DINO。
> v2 的第 1、2、4 步已实现：
> - **一套位姿**：IMU 预积分与零偏、连续时间轨迹、用最终轨迹重新去畸变，以及带版本号的位姿集合（pose set）；
> - **标定**：相机时间偏差、卷帘快门和外参；
> - **语言路线**：模型接口与合成真值实现，跨视角身份关联和身份级语言特征；
> - **测量级（第 4 步）**：平面约束的 LiDAR BA（与 IMU 因子联合优化）、多次观测融合且每点带 σ 的测量点云、正射影像上的亚像素矢量拟合，以及手持真值验收（导入、控制点 / 检查点、报告）。
>
> DA3 / 3DGS / 真实模型属于第 3 步，需要 GPU 服务器。

## C++ / Python 分工

以下算法**完整地**在 C++（`gr_core`）里实现，Python 只负责读写文件、把 YAML 配置映射成 C++ 配置结构（`core_config.py`）并按阶段调用：

| 模块 | C++ 入口 | 内容 |
|---|---|---|
| 点云配准 | `registration` | 体素哈希地图、多尺度点到面 ICP、由粗到精 ICP（`icp_multiscale`） |
| 激光里程计 | `imu`、`odometry` | 轮速 + 陀螺运动先验、扫描时间窗内 IMU 去畸变、逐帧配准、关键帧选择与关键帧点云（`OdometryPipeline`） |
| 回环检测 | `keyframes`、`loop_closure` | 位置半径 + 楼层高差门控、Scan Context 排序、子图多尺度 ICP 验证（按查询帧并行）、PCM 一致性筛选 |
| 位姿图优化（含 IMU） | `imu_preintegration`、`pose_graph`、`backend` | 入口 INS / GNSS 锚点（按时间戳用轮速 / 陀螺推到关键帧）；IMU 预积分因子（流形预积分，含零偏雅可比）、每个关键帧的速度与零偏状态、轮速 + 非完整约束的车体速度因子、零偏随机游走；LM 优化；整个阶段一次调用（`run_graph_stage`），以及逐帧位姿修正。加速度计重力因子、扫描到地图精修保留为可选项 |
| LiDAR BA | `plane_factor`、`lidar_ba`、`pose_graph`、`backend` | 自适应八叉树找各关键帧共有的平面（边角体素切分到不混面）；每个平面存各关键帧点的一、二阶矩，平面参数用 Schur 补消去（BALM 式代价 W·λ_min）；平面因子与 IMU、轮速、里程计、回环、锚点因子联合 LM 优化；位姿块稠密求解（先消去惯性状态），多轮"重新提取平面 + 优化"（`RunLidarBa`） |
| 连续时间轨迹（位姿集合） | `trajectory`、`exposure` | 以全部激光帧位姿为节点，节点间用去零偏陀螺和轮速积分并做端点修正（`ContinuousTrajectory`）；用它重新去畸变关键帧（`DeskewKeyframeSweep`）；相机每一行按曝光时刻取位姿（卷帘快门，`CameraExposure`） |
| 相机标定 | `camera_calibration` | 在固定轨迹上联合估计相机时间偏差与外参：多视图特征轨迹 + 三角化路标 + 路标到 LiDAR 面的点面约束，LM + 路标 Schur 消元（`CalibrateCamera`） |
| 地图与动态点判断 | `static_map`、`visibility` | 测量点云：体素内多次观测按距离加权融合（只取与最近观测距离相当的观测），每点带沿法向的 σ；法向定向、按趟距离图像可见性投票、自车扫过区域、高度门限，输出“本趟内移动 / 跨趟变化”（`DynamicDetector`） |
| 占据栅格 | `objects`、`occupancy` | 目标框时间线（停放 + 移动轨迹插值）、每帧语义占据与 LiDAR / 相机可见性掩膜（`OccupancyGenerator`） |
| 投影 | `projector`、`labeling`、`raster` | z-buffer 渲染、逐帧标签图、车位 / 标线 / 目标框投影与遮挡、投影一致性分（`FrameLabeler`）、2D→3D 多视角投票（`LabelLifter`）、地面纹理的遮挡渲染；全部按曝光（卷帘快门）投影 |
| 时钟同步 | `clock_sync` | 两个陀螺分窗口互相关（积分后的平均角速度，避免插值相位偏差），每窗口估旋转，稳健拟合偏差与漂移 |
| 与参考点云比较 | `cloud_compare` | 两个重力对齐点云的粗配准（墙面方向定航向的 4 个候选，立面占据图相关定平移，水平面高度直方图定高程），点到参考表面的距离（验收用） |
| 跨视角身份关联 | `identity` | 每帧 2D 掩膜经共享位姿和 LiDAR 地图投到 3D，按已识别点的归属加入身份或合并身份（同一帧的不同掩膜互斥），点按投票得到身份（`IdentityAssociator`，替代 ILGS 中的视频跟踪器） |
| 导航地图 | `nav_map` | 精确欧氏距离变换、Zhang-Suen 细化、骨架提图（节点聚类、毛刺剪枝、二度节点合并）、Douglas-Peucker 简化、Dijkstra 路网（含单行边）、8 邻域栅格 A* |
| 3DGS 支撑 | `gaussians`、`coverage` | 由 LiDAR 测量点生成扁平高斯（法向轴取点的 σ，切向取局部点距）；沿射线渲染高斯的深度和强度（3DGRT / 3DGUT 式逐高斯求峰值，网格 DDA 遍历），用于强度通道和留出扫描核验；相机覆盖统计（每点有效视角数、最佳分辨率、加权颜色）；训练掩膜（移动目标、跨趟变化物体、目标框） |

这些 C++ 接口不依赖 Python，可以直接在 C++ 程序里调用：`cpp/gr/e2e_test.cc` 在仿真车库里跑两趟，从里程计、图优化、地图、动态点一直到占据栅格和逐帧标签，全程不经过 Python。
Python 部分：
- 数据与标定读写、位姿集合（`recon/pose_set.py`）与产物版本记录（`provenance.py`）、阶段编排；
- 楼层 / 坡道与 BEV；
- 第五部分的语义规则（结构、标线、车位、目标检测跟踪、规则、导出）；
- 语言路线的特征提取与身份级语言特征（`semantics/language_field.py`）；
- 模型接口与合成真值实现（`models/`）；
- 评测、自检和合成数据。

## 目录

```
cpp/gr/                  C++ 核心库 gr_core：每个模块 <name>.h / <name>.cc / <name>_test.cc 放在一起，按 "gr/<name>.h" 引用
  types / pose / voxel / numeric / logging / check / image   基础类型、SE(3)、体素哈希、numpy 兼容数值、日志、GR_CHECK
  camera        针孔 / Kannala-Brandt 鱼眼 / Double Sphere
  registration  体素哈希地图 + 多尺度点到面 ICP（稀疏线束下退化为点到点会偏向“零运动”，见下文）
  imu           轮速 + 陀螺运动先验、IMU 去畸变、重力测量
  imu_preintegration   IMU 流形预积分（增量、协方差、零偏雅可比、残差与雅可比）
  odometry      KISS-ICP 风格激光里程计 + 每趟前端 OdometryPipeline（先验、去畸变、关键帧）
  keyframes     关键帧表（多分辨率点云、子图、累计里程）
  scan_context / loop_closure   回环：门控、Scan Context、多尺度 ICP 验证、PCM
  pose_graph / backend          LM 位姿图（相对位姿、先验、GNSS、重力；速度 / 零偏状态与 IMU、车体速度、零偏因子；平面因子），稀疏或稠密 Schur 求解；锚点、LiDAR BA、整段图优化
  plane_factor / lidar_ba       BA 平面因子（点簇矩、平面消元的高斯-牛顿项）、八叉树平面提取
  cloud_compare                 与参考（手持）点云的粗配准和点到表面距离
  clock_sync                    两个陀螺的时钟偏差与漂移（LiDAR 时间戳同步）
  trajectory    连续时间轨迹（节点 + 陀螺 / 轮速）、按轨迹去畸变
  exposure      相机曝光：每一行的位姿（卷帘快门）与自洽行投影
  camera_calibration   相机时间偏差与外参（多视图轨迹、点面约束、Schur 消元）
  static_map / visibility       地图累积（多次观测融合、每点 σ）、法向、按趟可见性投票的动态点判断、自车扫过区域
  objects / occupancy           目标框时间线、每帧语义占据与可见性掩膜
  projector / labeling / raster z-buffer、点云空间索引、逐帧标签与一致性分、2D→3D 投票、BEV 栅格与地面纹理
  identity      掩膜的跨视角身份关联（几何投影、合并、互斥约束、点投票）
  nav_map       导航地图：距离变换、细化、骨架提图、路网最短路、栅格 A*
  gaussians / coverage          3DGS 支撑：LiDAR 锚定高斯、沿射线渲染、相机覆盖、训练掩膜
  raycast / geometry / sim
  e2e_test.cc   仿真车库两趟的纯 C++ 全链路
  testing/      测试框架（GR_TEST / GR_EXPECT_*）与仿真场景、传感器
cpp/bindings/            pybind11 模块 garage_recon._core：bind_core（基础）、bind_recon（第四部分）、bind_labels（第五部分）、bind_nav（导航地图）、bind_splat（3DGS 支撑）
python/garage_recon/
  core_config.py  YAML 配置 → C++ 配置结构
  provenance.py   每个阶段产物所用的位姿集合版本（过期产物由自检拒收）
  data/       趟（trip）目录格式、标定（rig.yaml）、合成数据生成器 synthetic/（含 ground_truth.py：真值重渲染与点级真值；handheld.py：模拟手持扫描真值）
  models/     基础模型接口 interfaces.py（分割、语言编码、稠密特征、匹配、深度、OCR）、合成真值实现 oracle.py、按配置构建 registry.py
  recon/      第四部分：odometry → keyframes → backend（回环 + 图优化）→ trajectory / pose_set（位姿集合、重新去畸变）→ calibration（相机标定）→ static_map → levels → bev → floor_texture；stereo 接口
  semantics/  第五部分：structure → markings（vector_fit：正射影像上的亚像素矢量拟合）→ slots → objects4d → occupancy → frame_labels → language_field（features → identities → language）→ numbers（OCR 车位号）→ rules；lifting（2D→3D 投票）、adapters（真实模型适配器占位）
  nav/        导航地图：layers（分层）→ graph（路网、桥、坡道）→ goals（车位目标点）→ address（地址与查询）→ qa → export；navmap 编排
  eval/       对照合成真值评测：synthetic_eval、pose_eval（零偏 / 标定 / 曝光重投影）、semantic_eval（语言路线）
  acceptance/ 手持真值验收：handheld（读点云与标注）、scan_import（扫描仪导出的裁剪、抽稀与楼层 / 柱草稿）、registration（自动 / 控制点配准、柱检查点量测）、report（表面精度、完整度、检查点、矢量、结论）
  convert/    真实数据转换：profile（转换配置）、rosbag（ROS 1 / 2 bag → trip）、profiles/（FusionPortableV2 手持设备等）
  qa/         无真值自检（位姿集合与过期产物、平面厚度、BA 平面残差、点的 σ、回环与 IMU 残差、标定、多趟重复性、投影一致性、跨视角一致率、规则违例）
  io/         LAS 1.4 / PLY / GeoJSON / Lanelet2 OSM 导出；viewer（交互式网页查看器的导出与模板）
  cli.py      命令行
tests/                   pytest（端到端 + 单元）
```

## 构建

依赖：CMake ≥ 3.18、C++17 编译器、Eigen3、Python ≥ 3.9（numpy、scipy、pyyaml、opencv、scikit-image、shapely），可选 libomp（OpenMP）、laspy（LAS 导出）、rosbags（ROS bag 转换）。

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install pybind11 scikit-build-core laspy rosbags pytest
cmake -S . -B build -DPython_EXECUTABLE=$PWD/.venv/bin/python && cmake --build build -j8
./build/gr_tests                      # C++ 单元测试 + 纯 C++ 端到端（约 15 秒）
PYTHONPATH=python .venv/bin/python -m pytest -q   # Python 测试（含一次 tiny 端到端，约 40 秒）
```

开发模式下扩展模块直接生成到 `python/garage_recon/`；也可以 `pip install .`（scikit-build-core）安装成包（先删掉源码目录里的 `_core*.so`）。

C++ 代码遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)：函数 / 类型 CamelCase、变量 snake_case、类成员带下划线后缀，位姿命名为 `a_from_b`（如 `world_from_lidar`），`#define` 头文件保护，80 列，格式由仓库根目录的 `.clang-format`（Google 风格）统一：

```bash
find cpp \( -name '*.h' -o -name '*.cc' \) -print0 | xargs -0 clang-format -i
```

核心库不使用异常（以 `-fno-exceptions` 编译）：违反前置条件属于编程错误，由 `GR_CHECK` 报告后终止；可能“正常失败”的操作返回 `std::optional`。Python 绑定在调用核心库之前校验所有输入，并转换成 Python 的 `ValueError` / `IndexError`（pybind11 本身依赖异常，所以只有绑定层使用异常）。

新写和改动的 Python 模块遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)：
- 80 列；
- 只导入模块（`typing` 除外），不用相对导入；
- 模块、类和公开函数都有 docstring（Args / Returns / Raises）。

其余未改动的旧模块仍是原来的写法。

> macOS 注意：部分 Command Line Tools 安装只在 SDK 里带了 libc++ 头文件，clang 会报 `'vector' file not found`。CMakeLists 已自动检测并改用 SDK 中的头文件；重装 Command Line Tools 也能解决。

## 快速开始

```bash
# 一条命令：生成合成车库 → 重建 → 语义 → 自检 → 评测
PYTHONPATH=python .venv/bin/python -m garage_recon.cli demo --out out/tiny --preset tiny

# 分步（small = 两层 + 坡道 + 两趟）
PYTHONPATH=python .venv/bin/python -m garage_recon.cli synth --out data/small --preset small
PYTHONPATH=python .venv/bin/python -m garage_recon.cli recon --trips data/small/trip_00 data/small/trip_01 --work work/small --gt data/small/gt
PYTHONPATH=python .venv/bin/python -m garage_recon.cli semantics --trips data/small/trip_00 data/small/trip_01 --work work/small --gt data/small/gt
PYTHONPATH=python .venv/bin/python -m garage_recon.cli qa --trips data/small/trip_00 data/small/trip_01 --work work/small
PYTHONPATH=python .venv/bin/python -m garage_recon.cli eval --work work/small --gt data/small/gt
PYTHONPATH=python .venv/bin/python -m garage_recon.cli accept --work work/small --handheld data/small/handheld
```

- `--stages`：只跑部分阶段（各阶段可重复执行）。
- `--config my.yaml`：覆盖默认参数（`python/garage_recon/default_config.yaml`）。
- `--gt`：让模型接口使用合成数据的真值实现（`models.backend: auto`）。不给 `--gt` 且没有接真实模型时，相机标定和语言路线会跳过。
- `synth --ideal`：生成没有传感器误差的数据（无零偏、全局快门、时间偏差为 0、外参准确）。
- `synth` 同时生成模拟的手持扫描真值（`data/small/handheld/`，`--no-handheld` 跳过）；已有数据可用 `handheld --data DIR` 补生成。
- `accept`：把重建配准到手持真值并出验收报告（`work/small/acceptance/report.md`），见下文“测量级验收”。

## 输入：一趟数据的目录格式

```
<trip>/meta.yaml            trip_id、garage_id、可选 initial_pose（入口处 INS 位姿，world←vehicle）及其时间戳 initial_pose_time（缺省为第一帧 LiDAR 时刻）
<trip>/rig.yaml             LiDAR / 相机内外参（T_vehicle_sensor）、相机时间偏差与卷帘行时间、IMU 外参、轮速比例
<trip>/lidar/<name>/<ns>.npz   points (N,3) 传感器坐标、intensity、time（相对帧时间戳的秒数，≤0）
<trip>/cameras/<name>/<ns>.jpg
<trip>/imu.csv  wheel.csv  gnss.csv
```

相机时间约定：时间戳为 t 的图像，第 r 行在 t + time_offset + r × rolling_shutter_line_time 曝光。行时间来自传感器手册；时间偏差和外参由 calibration 阶段估计。

外部 LIO（如 FAST-LIO2 / FAST-LIVO2）的轨迹可通过 `odometry.external_trajectory` 接入，替代内置里程计。

## 真实数据：ROS bag 转换与扫描导入

### `convert`：ROS bag → trip

```bash
PYTHONPATH=python .venv/bin/python -m garage_recon.cli convert \
    --bag rec.bag --profile python/garage_recon/convert/profiles/fusionportable_v2_handheld.yaml \
    --out data/real/trip_00 [--start 0 --duration 60]
```

- 用纯 Python 的 `rosbags` 读 ROS 1 bag 和 ROS 2 bag（sqlite3 / mcap），不需要装 ROS（`pip install rosbags`）。
- **转换配置（profile，YAML）**写明：
  - 话题；
  - 各传感器相对一个参考传感器的外参：4×4 矩阵，或四元数 + 平移，`invert: true` 表示文件里写的是反方向；
  - 相机内参，以标定时的分辨率给出，可按 `scale` 缩小，内参随之缩放；
  - 各数据流到 IMU 时钟的时间偏差（相机的写进 rig.yaml，由标定阶段细化）；
  - 车体坐标系放在参考传感器的什么位置。

  见 `convert/profile.py` 的说明。
- **各数据流的转换**：

  | 数据 | 消息类型 | 处理 |
  |---|---|---|
  | LiDAR | `PointCloud2` | 按字段表解码，支持有组织点云和行填充；每点时间换成相对扫描结束时刻的秒数；按距离裁剪 |
  | IMU | `Imu` | 写成 imu.csv，保持 IMU 自身坐标系 |
  | 轮速 | `TwistStamped` / `Odometry` / `JointState` / `Float` | 写成 wheel.csv |
  | GNSS | `NavSatFix` | 转成局部 ENU 坐标 |
  | 相机 | `Image` / `CompressedImage` | 可降采样、限帧率 |
  | INS | `Odometry` | 取第一帧时刻的位姿作为入口位姿（`initial_pose` 与 `initial_pose_time`） |
- **不是车的采集平台**（手持、背包）没有天然在地面上的车体坐标系：
  - `vehicle.level: true`：用起步静止时 IMU 测到的重力把 z 轴摆正；
  - `vehicle.floor: auto`：用前几帧点云里参考传感器下方的地面（C++ RANSAC 平面）把原点放到地面上。

  流程依赖这两点：楼层与自车扫过区域假定车体原点在地面上；没有 INS 位姿时，世界坐标系就是起步时的车体坐标系，IMU 因子要求它重力对齐。
- **LiDAR 时钟按每段录制估计**（`lidars.<name>.clock_sync`）：
  - 用雷达自带的 IMU（与点云同一时钟）和主 IMU 的陀螺分窗口互相关（C++ `EstimateClockOffset`，每个窗口同时求两 IMU 间的旋转），逐帧修正点云时间戳；
  - 模型可选分段线性（`piecewise`，默认）或"偏差 + 线性漂移"（`linear`）；
  - `imu_delay` 是雷达 IMU 相对点云的固定延迟，属于雷达型号的常数（OS1-128：−22 ms）；
  - 估计结果写进 meta.yaml。

  这个偏差不是常数：FusionPortableV2 四段录制里，Ouster 时钟相对 STIM300 差 2–82 ms，一段 400 s 的录制里还漂移了 13 ms。
- 轮速还支持编码器累计脉冲（`kind: joint_position`，中心差分求速度）；`vehicle.reference_xy` 可把车体原点放到后轴（横向速度为零的点）。
- 自带 FusionPortableV2 手持设备和无人车的配置（`convert/profiles/fusionportable_v2_{handheld,ugv}.yaml`，标定取自官方 20230426_calib）。

### `import-scan`：扫描仪导出 → 手持真值目录

```bash
PYTHONPATH=python .venv/bin/python -m garage_recon.cli import-scan \
    --cloud merged_scan.pcd --trajectory scanner_path.tum --margin 15 --voxel 0.02 --out data/real/handheld
```

- **读入**：`handheld.read_cloud` 支持的所有格式。
- **裁剪**：按给定轨迹（扫描仪或车辆在点云坐标系里的路径）两侧 `margin` 范围裁剪，或按包围盒裁剪。
- **抽稀**：分块体素抽稀，用 C++ `voxel_downsample_indices`。
- **起草 `features.json`**：
  - 楼层：轨迹正下方地面高度的聚类；
  - 矩形柱：楼板以上 0.3–2.2 m 的高度带里，竖向覆盖完整、外形紧凑的 4 cm 栅格块作为候选。停放车辆高度不够或有玻璃，墙太长，都会被排除。每个面再用其附近点的中位数量测，得到柱心、尺寸和朝向。

  起草的要素都带 `"draft": true`，验收签字前需要人工复核。

## 结果导出与查看器（`viewer`）

```bash
PYTHONPATH=python .venv/bin/python -m garage_recon.cli viewer \
    --work work/a work/b --name "停车场" "车库" --out viewer --serve 8765
# 浏览器打开 http://localhost:8765/
```

- **导出内容**：每个重建目录导出成一个数据集。
  - 地图点：体素抽稀到最多 `--max-points`，坐标量化到 1 cm；
  - 每点属性：反射强度、趟次、语义类别、动态 / 跨趟变化、σ、离地高度；
  - 各趟轨迹；
  - 车位、标线、柱和墙的矢量；
  - 另附全分辨率的 `files/map.ply`（带全部属性）和各趟 `TUM` 轨迹，可在 CloudCompare 等工具里打开。
- **查看器**：three.js 加 GPU 着色器，数百万点也流畅。
  - 操作：左键拖动旋转，右键或 Shift + 左键平移，滚轮在光标处缩放，双击把点击处设为旋转中心；另有俯视 / 斜视 / 复位按钮。
  - 着色：高程、反射强度、趟次、语义类别、点位精度 σ、动态 / 变化。
  - 离地高度上下限可以去掉顶板，看清车库内部。
  - 图层：轨迹、车位、标线、柱与墙，可分别开关。
- **打开方式**：要通过 HTTP 打开（`--serve`，或任意静态文件服务），浏览器不允许直接从本地文件读取数据。

## 第四部分：车载重建

| 阶段 | 做法 | 输出（work/） |
|---|---|---|
| odometry | 逐帧点到面 ICP（约 15 ms/帧）；扫描时间窗内的陀螺 + 轮速逐点积分去畸变，同时作为配准初值；按距离/角度选关键帧并保存去畸变点云 | `trips/<trip>/odometry.tum`、`kf/` |
| graph | 位姿图 = 各趟里程计链 + 回环 + 入口锚点（INS 位姿 / GNSS，按各自时间戳推到关键帧）+ **IMU 预积分因子**（每个关键帧带速度与零偏）+ 轮速车体速度因子 + 零偏随机游走；回环候选按位置半径和楼层高差门控，Scan Context 排序，多尺度 ICP 验证，PCM 剔除不一致回环；最后做 **LiDAR BA**：30 m 内的回波切成八叉树平面（约 1 万个，各 ≤ 20 个关键帧），平面因子与上述全部因子联合优化，2 轮 | `graph/`（含每个关键帧的速度与零偏、BA 每轮统计）、`trips/<trip>/poses.tum` |
| trajectory | 生成**位姿集合**：每趟一条连续时间轨迹（全部激光帧位姿为节点，节点间用去零偏陀螺和轮速），加上 rig 标定，带内容哈希 `pose_set_id`；用最终轨迹重新去畸变全部关键帧点云 | `pose_set/`、`trips/<trip>/kf_final/` |
| calibration | 每台相机：锚点图像的查询点匹配到后续若干帧（DINO / RoMa 类匹配器，合成数据用真值实现）形成多视图轨迹，在固定轨迹上联合估计时间偏差和外参（路标约束在 LiDAR 面上）；不达标的相机保留 rig.yaml；结果写成新版本的位姿集合 | `calibration.json`、`pose_set/` |
| map | 累积全局测量点云（5 cm，只取 20 m 内的回波；同一体素内与最近观测距离相当的观测按 1/σ² 加权平均，σ 随距离增大；每点带沿法向的 σ 与观测次数）、法向；楼层/坡道识别（以车辆轨迹为主，每层地面高度栅格取中值并排除坡道）；动态点：距离图像可见性投票，按趟统计，区分“本趟内移动”和“跨趟变化”，再加自车扫过区域必为空闲的约束 | `map/point_map.npz`、`levels.json` |
| bev | 每层地面强度、地面高度、物体高度栅格 | `bev/level_<i>.npz` + PNG |
| texture | 四路鱼眼投影到地面网格（每一行按曝光时刻取位姿），每格取最佳视角（近、俯视、未被遮挡），中值融合去高光 | `bev/level_<i>_rgb.png` |

第三步之后的所有阶段只从位姿集合取位姿，并在 `provenance.json` 记录所用版本。只依赖 LiDAR / IMU 的阶段比较 `trajectory_id`，用到相机的阶段比较完整的 `pose_set_id`。自检会列出用旧版本生成、需要重跑的阶段。

关键经验（在合成数据上验证过）：

- **点到点 ICP 在稀疏线束下会偏向“零运动”**：地面、顶板、墙面上的扫描环在原位置对得最好。改为多尺度点到面，并跳过孤立的单环线段后，单次配准误差从 0.36 m 降到毫米级。
- **坡道相连的楼层需要重力约束**：没有 IMU 重力因子时，下坡时引入约 1° 的俯仰误差，整层地图倾斜，下层高程偏差约 0.3 m，回环修不掉。加入重力因子后，轨迹 ATE 从 17 cm 降到 3 cm。
- **转弯时的重力测量不可信**：车身侧滑使"轮速×角速度"的向心补偿不完整，转弯时重力方向误差 1–2°，会把横滚拉偏约 0.5°。只在不转弯时加重力因子，转弯段靠里程计衔接。
- **去畸变要用 IMU，不能用帧间平均角速度**：帧间平均角速度在转弯时与扫描期间的真实角速度差别很大。改用扫描时间窗内的陀螺积分后，里程计最大旋转误差从 3.5° 降到 0.7°。
- **地图只用近处回波**：掠射角下，0.3° 的姿态误差会让 25 m 外的地面点偏 10 cm 以上。限制在 20 m 内并让近处观测优先后，地图 p90 误差从 23 cm 降到 2 cm。
- **动态点要按趟、按目标判断**：逐点投票对移动车只覆盖约一半的点，后来再没被观测到的部分没有证据。按聚类整体判定后，移动车拖影能被完整清除。
- **停放车辆在地图层面判定**：多次经过的观测聚合后车身才完整。按车位拆分、向车位后端补全，比逐帧单视角拟合准确得多（中心误差约 10 cm）。
- **重力因子会被加计零偏带偏，IMU 预积分可以把零偏一起估出来**：0.065 m/s² 的加计零偏让重力方向偏约 0.4°，坡道相连的两层因此相对倾斜。IMU 因子带每个关键帧的速度和零偏后，零偏误差 < 0.002 m/s²。
- **IMU 噪声要用"等效值"**：噪声密度设成传感器白噪声时，IMU 的系统误差（仿真中是半个采样的时间错位和数值微分）会把轨迹拉弯，最大姿态误差 0.5°。陀螺 0.01 rad/s/√Hz、加计 0.2 m/s²/√Hz 时最好；放到 300 倍又会丢掉横滚 / 俯仰约束。实车按 Allan 方差加放大系数标定。
- **轮速的非完整约束要放宽**：车体坐标系与运动方向并不严格一致（转弯、坡道衔接、侧滑）。侧向 / 垂向速度 σ 从 0.05 放宽到 0.3 m/s 后，ATE 从 3.9 cm 降到 2.2 cm，地图 p90 从 3.4 cm 降到 2.0 cm。
- **相机时间偏差是"等效"参数**：1.4 m/s 时，1 cm 的轨迹沿线误差相当于 7 ms。用真值轨迹标定时无偏，用估计轨迹时会吸收轨迹误差；轨迹 ATE 6.8 cm 时偏差约 5 ms，2 cm 时 ≤ 2 ms。测量级相机几何仍需硬件同步。
- **平面约束的 LiDAR BA 修掉了层间偏移**：扫描到地图精修（每个关键帧对齐其余关键帧的地图）收敛到一个自洽但有偏的解。坡道两端地面与坡面的折线处，局部平面拟合把两个面混在一起，把坡度"拉平"约 0.05°，两层的相对高程差约 2 cm。BA 的八叉树把折线处的体素切分开，再与 IMU 因子联合优化：柱检查点的两层高程偏差从 −0.8 / +0.9 cm 降到 ±0.1 cm。
- **BA 让入口锚点的时间错误暴露出来**：地图变得刚性后，入口锚点的错误不再被局部吸收，而是把整张图拉偏：
  - 每趟第一帧没有去畸变：IMU 先验拒绝起点早于第一个 IMU 样本的区间，第一帧扫描恰好从那一刻开始，浮点舍入后差一点点。第一帧因此与地图错开 0.2°，INS 航向先验传不到地图上。
  - INS 位姿取自第一帧 LiDAR 前 0.1 s，却挂在第一帧上（沿行驶方向差 14 cm）；早于第一帧的 GNSS 定位也同样被钳到第一帧。
  - 修法：IMU 先验在边界外允许保持边界值 0.1 s；连续轨迹在首尾外推 0.2 s；INS 位姿带时间戳（`initial_pose_time`），INS 和 GNSS 都用轮速 / 陀螺先验推到关键帧时刻。
- **BA 要用远处的回波**：均匀的坡道通道在行驶方向上几乎没有约束。BA 只用 20 m 内的回波时，两层沿坡道方向错开约 2 cm；改为 30 m 后，刚体对齐后的轨迹误差从 1.0 cm 降到 0.4 cm。
- **测量点云要融合多次观测**：同一体素只留最近一次观测时，地图噪声就是 1 cm 的测距噪声。把距离相当的观测按 1/σ² 加权平均后，配准后的表面误差中位从 0.37 cm 降到 0.22 cm；每点的 σ 基本可信：78% 的误差在 1σ 内，93% 在 2σ 内。
- **BEV 的"物体高度"栅格曾把顶板算进去**：高度上限 3.5 m 而顶板在 3.3 m，凡是看到顶板的地面格都被当成"被物体遮挡"。结果是标线被大片遮罩，60 个车位有 59 个被判为"采集时有车"。现在只统计 2.2 m 以下的点。
- **矢量要在亮度图上重测**：骨架化的线端点每端缩进约半个线宽。在亮度剖面上找半亮度边缘和端点后，可靠端点误差约 1–2 cm。256 px 鱼眼拼出的正射影像比 LiDAR 强度图模糊（中心线 RMS 3–4 cm 对 1 cm），所以默认先用 LiDAR 强度；换成高分辨率相机或 3DGS 正射影像后应改为相机优先（`vector_fit.FitConfig.sources`）。
- **柱面要在点云上量**：栅格法的柱宽被量化成 0.55–0.6 m（真值 0.5 m），柱面作为车位虚拟边线时偏 7–16 cm。在地图点上拟合各柱面后，柱心误差从约 10 cm 降到 0.9 cm。
- **测量级精度要在控制点配准后评**：车辆坐标系的绝对位置只由入口锚点决定，误差约 1–2 cm、0.03°。合成数据评测不做配准，会把这部分算进地图误差；验收按测量惯例先用控制点定坐标系，再看检查点。
- **跨视角身份关联必须做背面剔除**：遮挡测试为照顾掠射角地面而放宽后，楼板另一面（上层地面）会被当成可见，和下层顶板合成一个身份。用地图法向剔除背面、按像素足迹 / cos 给掠射角容差后，身份纯度从 0.79 升到 0.90。

## 第五部分：自动语义标注

| 阶段 | 做法 | 输出（work/semantics/） |
|---|---|---|
| structure | 地面/坡道/顶板/机电按高度与法向标注；柱：多高度切片都占据且到达最高切片的紧凑块，碎片合并，再在地图点上拟合各柱面（法向朝外的点的中位位置），看不到的一侧按典型柱宽补全，按柱网聚类生成轴号；墙：沿网格方向的直线段，要求薄、且到顶或长于一辆车（避免把车侧面当墙） | `labels.npz`、`structure.json` |
| markings | 每层 LiDAR 强度图 + 鱼眼地面纹理：分层白顶帽、稳健阈值、沿网格方向先闭后开（连接激光环采样的虚线）、骨架化、拟合直线、共线合并；要求线两侧都是已观测地面且与网格对齐。然后在正射影像（亮度图）上**亚像素重测**每条线：横向剖面的半亮度边缘定中心线（稳健直线拟合），沿线的半亮度交点定端点；端点外侧地面被观测到才算"可靠端点"（车、墙根、柱旁的端点不可靠） | `markings.json/.geojson` |
| slots | 隔线分排、排内相邻隔线成车位；每排按隔线端点估计自身方向（排与柱网常差 1–2°），在排自己的坐标系里组装；被车挡住的隔线按整数倍宽度推断，但推断要有理由：假想隔线落在看得清、没有车的地面上且强度图上没有油漆脊线，就不推断；车辆开过、与有线车位重叠、或大部分落在未观测区域（墙后）的推断车位也去掉；柱面作为虚拟隔线；背靠背两排按隔线端点的 10 / 90 % 分位判断后对半拆分；排的入口线 / 后端线取各隔线可靠端点的中位数，看不到的一端按实测车位深度补齐；入口朝向行驶过的车道；记录采集时是否有车 | `slots.json/.geojson` |
| objects | 停放车辆在聚合地图上判定（多视角拼完整、按车位拆分与补全、跨趟变化的车只归属对应那一趟）；移动目标逐关键帧检测、匈牙利匹配跟踪、轨迹拼接、轨迹级平滑与尺寸估计 | `objects/static.json`、`moving.json`、`tracks.json` |
| occupancy | 每帧自车坐标系语义占据（静态标签 + 本帧目标框），LiDAR / 相机可见性掩膜 | `occupancy/<trip>/<ns>.npz` |
| frames | 每张鱼眼图（每一行按曝光时刻取位姿）：静态地图 + 目标框的 z-buffer → 语义标签图；车位角点/入口线、标线、目标框投影并做遮挡判断；投影一致性分（投影的标线边缘是否落在图像边缘上）；记录 `pose_set_id` | `frames/<trip>/<cam>/<ns>.json` + `_label.png` |
| features | 语言路线第一步：每张图像的类别无关掩膜（Segmenter2D：SAM 2 / SAM 3）和每个掩膜的语言嵌入（LanguageEncoder：SigLIP 2 / CLIP） | `features/<trip>/<cam>/<ns>.npz` |
| identities | 掩膜经位姿集合和静态地图投到 3D（C++ `IdentityAssociator`）：加入已有身份、合并同一物体的碎片、同一帧不同掩膜互斥；每个地图载体点得到身份。替代 ILGS 的视频跟踪器 | `identities.npz` |
| language | 每个身份的嵌入 = 各视角掩膜嵌入的加权平均（近、未被图像边界截断的视角权重高），与分类表中每类的中英文提示词比较得到类别；地图点继承身份的类别；再把 3D 标签渲染回每张图，与该图自己的 2D 预测比较得到跨视角一致率 | `language/labels.npz`、`summary.json` |
| numbers | 车位号 / 分区号：OCR（`models.ocr`）读出的每段文字，按其像素中心的射线落到各层地面，归给附近的车位投票；每排再按"同排编号连续"修正；"C-105"拆成分区 C 和编号 105 | `numbers.json`（`slots()` 读取时合并） |
| rules | 车位尺寸、编号重复、入口远离车道、推断车位、柱补全等进入人工审核队列 | `issues.json` |
| nav | 导航地图，见下一节 | `nav/` |
| export | LAS 1.4（类别码 64+id，附楼层/动态标志）、PLY、GeoJSON、Lanelet2（parking_space / parking_lot，带车位号 `ref` 和分区 `zone`） | `export/` |

模型接入点：
- **基础模型接口**（`models/interfaces.py`）：Segmenter2D、LanguageEncoder、DenseFeatures、Matcher、DepthPrior、OcrReader；
- **合成真值实现**（`models/oracle.py`）：从真值渲染派生，可设置类别混淆、嵌入噪声、过分割、像素噪声，用来在没有 GPU 和模型权重时验证整条链路；
- **其他接入点**：`semantics/adapters.py`（真实模型适配器占位、3D 检测器）、`recon/stereo.py`（学习式双目）、`semantics/markings.py:MarkingSegmenter`；OCR 读数由 `rules.fix_slot_numbers` 按“同排编号连续”修正。

几何规则路线不依赖深度学习，仍是默认产出。

## 导航地图（`nav` 阶段）

把语义结果整理成导航能直接用的地图：按地址找车位、给出目标位姿、规划不穿墙的全局路径。几何算法（距离变换、骨架、路网提取、最短路、栅格 A*）在 C++（`nav_map`），编排在 `python/garage_recon/nav/`。

**分层（每层 0.1 m 栅格）**
- **障碍按几何判定，不只看类别**：离地 0.12 m 到车高 + 0.2 m 之间的静态点都算障碍，墙被错分成别的类也照样挡住。无类别的点每格累计观测至少 6 次才算（移动目标残留、噪声通常只被观测一两次）。
- 再按类别分成固定结构（墙、柱、护栏、未分类物体）、悬挂结构（管线、梁、标志，低于车高时才算）、采集时的停放车辆、可越过的低矮物（减速带、挡轮器）。
- 可行驶区 = 观测到的地面 + 车辆开过的区域 − 障碍；四周都已观测到、面积小于 20 m² 的未观测地面按地面处理（被车挡住或线束间隙），大片未知区域仍算障碍。
- 输出 ROS map_server / nav2 的 `level_<k>.pgm/.yaml`（固定障碍为占据），各层所有图层 `level_<k>_layers.npz`，彩色预览图。

**路网**
- 每层取车道区（可行驶、非车位、非坡道）中与障碍保持"半车宽 + 0.3 m"间距的部分，细化成骨架，再提取节点和边。路网上每一点都留有这个间距，沿路网的路线不会穿墙、擦柱。
- 误检的车位或个别窄点会把骨架截断：同一层断开的几块用栅格 A* 补"桥"（可以穿过车位、只要求车身半宽），桥在 QA 里列出供复核；真正过不去的窄点保持断开。
- 每条坡道生成一条从坡顶到坡底的边，两端用直连线接到上下两层的路网，直连线经过障碍层检查。

**车位目标点（`goals.json`）**
- 停车位姿：车位中心，车头进与倒车进两个朝向；
- 接近点：从车位入口沿车道方向找到的车道中线点，带车道朝向；
- 入口外 0.3 m 的车位节点经直连线接入路网（只检查墙、柱等固定障碍，不看采集时的停车）；
- 从入口（默认第一趟的起点，可配置 `nav.origin`）出发的路线长度，以及是否可达。

**地址**：楼层名（`nav.level_names`，例如 `[B1, B2]`）、分区（OCR，或配置的分区多边形 `nav.zones`）、车位号（OCR）。查询支持"B2 C区 105号""b2-c-105""负二层C区105""C105"或车位编号（`nav/address.py`）。没有车位号时地址退回车位编号。

**QA（`qa.json`）**：从入口到每个车位按路网规划并逐段检查：碰到固定障碍的路线数（应为 0）、路线上的最小间距、不可达车位（用栅格 A* 复核，分为"路网断开"和"地图上本来就过不去"）、桥、开过的车位、车位内的障碍、采集轨迹落在路网上的比例。

**查看器**：`viewer` 导出时带上路网和目标点；页面里可以输入地址或车位编号，在浏览器里从入口规划并画出路线和停车朝向。

| 数据 | 车位 | 可达 | 车位号 | 碰到障碍的路线 | 路线最小间距 | 桥 |
|---|---|---|---|---|---|---|
| 合成 small（两层 + 坡道，真值 OCR） | 60 | 60 | 60（分区 + 编号全部正确） | 0 | 1.25 m | 0 |
| FusionPortableV2 室外停车场（3 趟，无 OCR） | 81 | 79 | 0 | 0 | 1.27 m | 4 |
| FusionPortableV2 地下车库（手持，无 OCR） | 37 | 36 | 0 | 0 | 1.20 m | 16 |

- 合成数据上路网就是车道中线，两层经坡道连通。
- 室外停车场：车位检测修正前，300 个车位里 223 个是推断的，大多铺进了车道或垂直于真实车排；按上面的推断规则和逐排方向修正后为 81 个，与地面画线吻合，仍有部分排因隔线碎裂漏检。开阔场地的骨架不像车道，更像一张网；按车位排生成车道线是下一步。
- 地下车库：去掉墙后的推断车位后剩 37 个；走廊里有几处只有约 2 m 宽的窄点，补成了桥并列在 QA 里。
- 两份真实数据都没有跑 OCR（需要下载 OCR 模型），所以地址暂时是车位编号。

## 合成数据上的评测结果

Apple M2（8 核）上的实测结果，数据为 small：两层 + 坡道、2 趟（停放车辆不同，各有一辆移动车）、32 线、256 px 鱼眼。

合成数据默认带传感器误差：
- 每趟 IMU 零偏：陀螺约 0.05°/s，加计约 0.065 m/s²；
- 相机卷帘快门，读出 25 ms；
- 相机时间偏差 6–16 ms；
- rig.yaml 外参误差 0.3° / 15 mm。

`synth --ideal` 生成没有这些误差的数据。模型接口都用合成真值实现（`--gt`）。

### 第 4 步（2026-10-13）

同一份有误差的数据，对比第 1–2 步的算法（扫描到地图精修、每个体素只留最近一次观测、骨架法矢量、栅格法柱）和第 4 步（LiDAR BA、多次观测融合、正射影像上的亚像素矢量拟合、点云上拟合柱面）。两边都已包含本轮修掉的错误：入口锚点的时间戳、第一帧去畸变、BEV 物体高度栅格。手持真值见“测量级验收”，控制点为每层柱网四角的柱，其余柱作检查点。

| 指标 | 第 1–2 步 | 第 4 步 |
|---|---|---|
| **对照合成真值（同一坐标系，不配准）** | | |
| 轨迹 ATE（刚体对齐后） | 2.2 cm（1.3 cm） | 1.6 cm（0.4 cm） |
| 旋转误差 均值 / 最大 | 0.035° / 0.16° | 0.014° / 0.10° |
| 地图到真值表面 中位 / p90 | 1.0 cm / 1.8 cm | 0.7 cm / 1.3 cm |
| 柱 召回 / 中心误差 | 32/40 / 4.6 cm | 35/40 / 1.7 cm |
| 标线 召回 / 精度 | 68% / 98% | 67% / 99% |
| 车位 召回 / 角点误差 均值（p95） | 59/60 / 4.3 cm（12.0 cm） | 60/60 / 2.2 cm（4.5 cm） |
| 停放车辆 F1 | 0.94 | 0.96 |
| 真值点重投影误差 均值 | 0.71 px | 0.56 px |
| 语言路线 点级准确率 / mIoU | 0.84 / 0.66 | 0.85 / 0.68 |
| **手持真值验收（控制点配准后）** | | |
| 控制点残差 RMS | 1.9 cm | 0.4 cm |
| 检查点（柱）RMSE 3D / 平面 / 高程 | 1.6 / 1.2 / 1.0 cm | 0.4 / 0.4 / 0.1 cm |
| 检查点 CE90 / LE90 | 2.1 cm / 1.2 cm | 0.5 cm / 0.2 cm |
| 两层的高程偏差（柱检查点） | −0.8 / +0.9 cm | +0.1 / −0.1 cm |
| 表面误差 中位 / p95 / 2 cm 内 | 1.0 cm / 2.5 cm / 91% | 0.2 cm / 1.6 cm / 97% |
| 误差在 1σ / 2σ 内（点的 σ） | 56% / 93%（仅测距模型） | 79% / 94% |
| 车位角点 RMS / CE90 | 5.7 cm / 7.5 cm | 1.7 cm / 2.5 cm |
| 柱心（语义检测）RMS | 7.9 cm | 2.0 cm |
| 标线 召回 / 中心线偏移 RMS | 68% / 2.4 cm | 67% / 1.7 cm |
| 地面完整度（观测区域内 10 cm） | 99.3% | 99.3% |
| 结论（检查点、表面、车位角点、完整度） | 3/4 通过（车位角点未过） | 4/4 通过 |
| 总耗时 | 约 9.8 分钟 | 约 9.4 分钟 |

几点说明：

- **两层高程偏差**是扫描到地图精修的系统误差（见"关键经验"），BA 后消失；检查点 RMSE 从 1.6 cm 降到 0.4 cm。
- **车位角点**：第 1–2 步的误差主要来自线端点（骨架缩进约半个线宽、墙根和车旁的端点看不到）和柱面（栅格柱宽量化为 0.55–0.6 m），不是此前以为的层间偏移。
- **绝对坐标**：两边的不配准 ATE（1.6 / 2.2 cm）都包含入口 INS / GNSS 锚点的误差；测量级精度看控制点配准后的检查点。
- **标线召回约 67%**：其余标线在两趟采集中都被停放车辆盖住，车辆看不到。
- **完整度**：地面 99.3%；中部高度带（墙、柱、车）约 94%，其中包括两次采集之间变化的停放车辆；顶部（顶板，车载 LiDAR 只能掠射）约 96%。
- **耗时**：图优化从 48 s 降到 18 s（BA 2 轮），地图融合多 1 s；其余阶段不变。

### 第 1、2 步（2026-10-12）

下表在同一份数据上对比 v1（第 1、2 步之前的代码：重力因子后端、按图像时间戳插值相机位姿、rig.yaml 外参、全局快门）和 v2。这些数字测于本轮修正入口锚点时间戳之前。

| 指标 | v1 有误差 | v2 有误差 | v1 理想 | v2 理想 |
|---|---|---|---|---|
| 轨迹 ATE（Umeyama 对齐后） | 3.6 cm（1.7 cm） | 2.2 cm（1.4 cm） | 2.9 cm（1.4 cm） | 2.2 cm（1.4 cm） |
| 旋转误差 均值 / 最大 | 0.162° / 0.28° | 0.061° / 0.16° | 0.089° / 0.18° | 0.062° / 0.16° |
| 地图到真值表面 中位 / p90 / 5 cm 内 | 1.4 cm / 4.4 cm / 93.0% | 0.9 cm / 2.0 cm / 99.6% | 0.7 cm / 2.1 cm / 98.5% | 0.9 cm / 2.0 cm / 99.6% |
| 柱 召回 / 中心误差 | 31/40 / 7.1 cm | 33/40 / 6.3 cm | 34/40 / 7.1 cm | 34/40 / 7.3 cm |
| 标线 召回 / 精度 | 68% / 100% | 69% / 100% | 69% / 100% | 69% / 100% |
| 车位 召回 / 角点误差 均值（p95） | 59/60 / 5.6 cm（9.4 cm） | 59/60 / 4.7 cm（13.1 cm） | 60/60 / 6.0 cm（15.3 cm） | 59/60 / 5.1 cm（13.1 cm） |
| 停放车辆 F1 | 0.94 | 0.96 | 0.96 | 0.94 |
| 移动车辆（每趟） | 1/1 | 1/1 | 1/1 | 1/1 |
| 真值点重投影误差 均值（p95） | 2.11 px（6.1 px） | 0.90 px（2.7 px） | 1.14 px（3.6 px） | 0.92 px（2.7 px） |
| IMU 零偏误差 陀螺 / 加计 | – | 0.0027°/s / 0.0010 m/s² | – | 0.0026°/s / 0.0005 m/s² |
| 相机外参误差 旋转 / 平移 | 0.3° / 15 mm（rig.yaml） | 0.013° / 4.1 mm | 0 | 0.015° / 3.3 mm |
| 相机时间偏差误差 | 10–14 ms | ≤ 1.4 ms | 0 | ≤ 1.1 ms |
| 语言路线 点级 准确率 / mIoU（规则路线） | – | 0.84 / 0.67（0.83 / 0.74） | – | 0.87 / 0.69（0.83 / 0.74） |
| 身份纯度 / 跨视角一致率 | – | 0.90 / 0.85 | – | 0.92 / 0.86 |
| 逐帧 mIoU：3D 渲染 / 单视角 2D | – | 0.62 / 0.69 | – | 0.65 / 0.70 |
| 总耗时 | 约 5.8 分钟 | 约 10.3 分钟 | 约 4.5 分钟 | 约 8.7 分钟 |

几点说明：

- **v2 的耗时构成**：第四部分约 3.5 分钟（其中相机标定 81 s、图优化 50 s、地面纹理 50 s）；第五部分约 6.5 分钟（其中逐帧标签 120 s、语言路线约 200 s）。
- **语言路线的合成真值实现**带 10% 的类别混淆和嵌入噪声。3D 身份聚合消除了大部分混淆，点级准确率高于规则路线，mIoU 仍低于规则路线：
  - 相机看不到的面（车顶等）没有 2D 依据，车辆点约 25% 未标注；
  - 地面细线的掩膜很小，车位线 IoU 为 0。
- **逐帧标签用点云渲染**，边界比 2D 掩膜粗，所以逐帧 mIoU 低于单视角 2D 结果。
- **车位角点 p95 偏大**：当时以为来自层间偏移；第 4 步查明主要是线端点和柱面的量测误差（见上节）。
- **漏检的柱子**都贴在坡道护栏 / 围墙上，只有贴墙的一面能被看到，几何上与墙无法区分，需要图像或 BIM 先验。
- **tiny 预设**（单层、1 趟、16 线、128 px）用于 Python 测试，整条链路约 40 秒。

## 测量级验收（手持真值）

方案第 3.5 / 7 节的验收：把车载重建配准到手持真值的坐标系，再按门槛出报告。

手持真值目录（扫描仪导出 + 在其上的标注）：

```
handheld/cloud.(ply|las|laz|pcd|xyz|txt|csv|npz)   手持点云，自身坐标系，重力对齐（z 向上）
handheld/features.json   可选：在手持点云上数字化的 levels / columns / slots / lines
handheld/points.csv      可选：测量的点（全站仪靶标等）：id, role(control|check), x, y, z[, map_x, map_y, map_z]
```

合成数据的手持真值（`data/synthetic/handheld.py`）：沿车辆路线每 2 m 一个扫描位置、1.5 m 高的近球形扫描（0.5° × 1°），3 mm 测距噪声，4 cm 体素平均；另一批停放车辆（不同时间采集）；手持 SLAM 误差（万分之一比例误差、每层 2 mm 高程漂移）；坐标系相对世界系旋转 23.7° 并平移约 160 m。真实变换只写在 `gt/handheld.json`，验收工具不用它。

`accept` 的流程：
1. **配准**。粗配准（C++ `align_gravity_aligned`）+ 多尺度点到面 ICP（地图结构点 → 手持点云），然后：
   - `control`（默认）：按测量惯例只用控制点定坐标系。有 `points.csv` 里带地图坐标的控制点时用它们；没有时每层取柱网四角的 4 根柱作控制点，在地图上量测柱心后做刚体拟合。其余柱作检查点。
   - `auto`：直接用 ICP 结果（整张地图的最佳刚体拟合）。
2. **检查点**：每根柱在地图点上量测（四个柱面的点 → 柱心 x / y，柱周地面 → 高程），与手持标注比较：3D / 平面 / 高程 RMSE、CE90、LE90、平均误差向量（系统偏差），分层统计。
3. **表面精度**：地图结构点到手持表面（局部平面）的距离：中位、p90、p95、2 cm 内比例，分层、按来源标记（L / G / P）；水平面的有符号偏差即高程偏差；点带 σ 时统计误差落在 1σ / 2σ 内的比例（σ 是否可信）。
4. **完整度**：手持点中被车辆观测到的（0.5 m 内有地图点）有多少被地图在 5 cm 内复现。
5. **矢量**：车位角点（按 IoU 匹配）、语义检测的柱心、标线中心线，都在手持坐标系里比较。
6. **结论**：门槛取方案第 7 节的初定值（`acceptance.thresholds`）：检查点 RMSE ≤ 2 cm、表面 p95 ≤ 3 cm、车位角点 RMS ≤ 2 cm、完整度 ≥ 90%。

输出 `work/acceptance/report.json`、`report.md`（中文报告）和每层的偏差图 `deviation_level_<i>.png`。真实项目只需把手持结果放成上面的目录（标注可由人工在手持点云上数字化），其余不变。

## 真实数据试跑：FusionPortableV2 地下车库（2026-10-14）

### 数据

- **数据集**：[FusionPortableV2](https://fusionportable.github.io/dataset/fusionportable_v2/) 的 `handheld_underground00`，港科大（广州）地下车库，404 s。
- **设备**：手持多传感器，没有轮速。Ouster OS1-128、STIM300 IMU、双目 1024×768 相机。
- **路线**：约 400 m 单程直走廊，没有回到走过的地方，所以没有真正的回环；起步时设备放在地上，静止约 20 s 后拿起。
- **真值**：
  - Leica 地面扫描合并点云（9564 万点）；
  - Leica MS60 跟踪仪轨迹，只有棱镜位置。
  - 两者不在同一坐标系，数据集也没给两者之间的变换。

### 流程

1. `convert`：用 `fusionportable_v2_handheld.yaml` 转换，约 3 分钟。
2. `recon` 和 `semantics`：用 `fusionportable_v2_handheld_recon.yaml`。
3. `accept --method auto`：把地图配准到整张真值点云（4 cm 抽稀）上。
4. 把我们的路径变换到真值坐标系，再用 `import-scan` 按路径裁剪、起草真值要素。

### 真实数据暴露的问题及处理

| 问题 | 现象 | 处理 |
|---|---|---|
| LiDAR 与 IMU 时间偏差 | 点云角速度与 STIM300 陀螺的互相关在 +37 ms 处最好（RMS 16.2 → 9.8°/s）；Ouster 内置 IMU 比 STIM300 晚 74 ms，其中约 37 ms 是它自身的滤波延迟。官方标定文件里没有这一项 | 当时在配置里手填 `time_offset: 0.037`，IMU 因子残差中位数从 12.1 降到 3.7；后来发现偏差随录制变化、还会漂移，改为转换时按段估计（`clock_sync`），见下一节 |
| 加速度计零偏和倾斜分不开 | 直线行走时水平零偏与倾斜不可区分；默认（消费级）先验 0.2 m/s² 下，零偏被估成 0.5 m/s²，整条路线倾斜 3%，楼层高度变成 −12 m | 战术级 IMU 先验收紧到 0.03 m/s² |
| 平面 BA 把走廊压弯 | 有无 IMU 因子都会压弯，竖向 ±2 m；单帧里地面随距离系统性偏高（15–20 m 处高 3–7 cm，四个方向都一样）。缩短 BA 量程、新增的按距离和入射角降权（`ba.angular_sigma`）都没用 | 这台设备暂时关闭 BA，待 LiDAR 内参标定 |
| 起步放在地上 | 只看开头几帧找不到车体下方的地面 | 地面高度改为在整段录制中抽 20 帧取中位数（1.26 m） |
| 没有轮速 | 只能靠纯匀速模型 | 新增陀螺运动先验 |
| 只有针孔相机 | 逐帧标签只处理鱼眼相机，导致崩溃 | 所有相机都出标签 |
| 真值里没有独立柱 | 两侧是凹进去的车位区，`import-scan` 正确地起草出 0 根柱 | 柱检查点和控制点在这里不适用 |

### 结果

| 指标 | 数值 |
|---|---|
| 轨迹对跟踪仪，刚体对齐，364 s / 408 m | 纯里程计 ATE 0.50 m（高程 0.39 m）；IMU 图优化 + 回环 ATE 0.42 m，其中高程 0.04 m、水平 0.41 m |
| 一次刚体配准到整张真值点云 | ICP 残差 4.5 cm；地图表面误差 p95 21.9 cm；地面完整度 71% |
| 按 40 m 分段各自配准（局部精度） | 表面误差中位约 2.5 cm，p95 约 9 cm |
| 验收结论 | 未通过：没有检查点；表面 p95 远超 3 cm |

- **水平误差**：0.41 m 来自 400 m 单程直走廊里不可观测的航向漂移。没有回环、控制点或 GNSS，任何纯相对方法都会有这种漂移。
- **局部精度不到测量级的原因**：
  - BA 关闭；
  - Ouster 的测距噪声；
  - 地面扫描时与手持采集时停的车不同；
  - LiDAR 的系统误差。

## 真实数据试跑：FusionPortableV2 室外停车场无人车（2026-10-14）

### 数据

- **序列**：`ugv_parking00`、`02`、`03`，同一室外停车场，三趟。
- **平台**：阿克曼小车，传感器与手持那套相同，另有后轮编码器。
- **路线**：车速 1.5–3 m/s，每趟都有回环。
- **真值**：
  - INS 轨迹（6 自由度），INS 自报不确定度 7–15 cm，RTK 浮点解和固定解混合；
  - Leica 校园点云，点距约 8 cm，拍摄时间不同。

### 数据上测出的标定

| 项目 | 结果 | 依据 |
|---|---|---|
| Ouster 内置 IMU 相对点云的延迟 | −22 ms | 三趟数据的时钟偏差相差 45 ms，但都在 −22 ms 处地图最薄、IMU 残差最小 |
| 车体原点 | 移到后轴（IMU 后方 0.98 m） | 转弯时的横向速度与转速成正比 |
| 轮速比例 | 1.022 | 与 INS 速度对比 |

### 结果（三趟合并）

| 指标 | 带 BA | 不带 BA |
|---|---|---|
| 三趟两两重复精度（重复观测的平面）：中位 / p95 | 0.64–0.69 cm / 约 1.9 cm | 1.6–2.8 cm / 5–20 cm（三趟没有完全合上，表面"重影"） |
| 趟间地面高差 | 小于 0.1 cm | 最大 1.6 cm |
| 合并后地图点数 | 234 万 | 488 万 |
| 对照 Leica 点云：地面误差中位 / p95 | 5.1 / 19.9 cm | 6.1 / 20.5 cm |
| 对照 INS 轨迹：ATE | 23–54 cm | 24–50 cm |

- 单趟里程计地图的平面厚度中位约 0.7 cm。
- 绝对精度受参照本身限制，验证不了厘米级：
  - INS 自报 7–15 cm；
  - 地面扫描点距 8 cm，地面有植被，拍摄时间不同。
- 这里 BA 明显有用（多趟数据、大量回环）；地下单程长走廊里 BA 反而有害，见上一节。

## 已知限制 / 下一步

- 导航地图：
  - 开阔场地的路网是可行驶区的中轴线，像一张网而不像车道；应改为按车位排的入口线偏移出车道中线，再与中轴线合并；
  - 单行方向未知（地面箭头尚未识别），所有边都是双向；车库出入口、限高标志、车位属性（充电、无障碍、专用）还没有；
  - 车位占用只是采集时刻的，导航要接实时状态；
  - 真实数据还没跑 OCR（需要下载 OCR 模型），地址退回车位编号；
  - 真实场地的车位检测仍有漏检（隔线碎成多段时整段丢失）。
- 3DGS（方案 3.4）：C++ 部分已完成（LiDAR 锚定高斯、沿射线渲染深度与强度、相机覆盖、训练掩膜、自由空间栅格）；训练（gsplat，需要 CUDA）、冻结与保护策略、强度损失、核验流程还没做。

- 真实数据（FusionPortableV2）上的待办：
  - 标定 LiDAR 内参（线束仰角）后重新打开 BA；
  - 没有雷达内置 IMU 时，在后端把 LiDAR–IMU 时间偏差作为待估参数（有内置 IMU 时已在转换时按段估计）；
  - 找一份有厘米级参照（全站仪控制点或密集地面扫描）的车载数据，验证绝对精度；
  - IMU 噪声和零偏先验改为按传感器等级配置；
  - 没有柱的车库需要别的检查点，例如墙角、车位角，或全站仪靶标（`points.csv`）；
  - 长走廊需要控制点或回环来约束航向。

- 合成路线的转弯很急（每帧最多 45°），转角处的单帧旋转误差可达 1–3°。真实 10 Hz 数据的转角误差会小得多。
- 最耗时的是逐帧标签、语言路线（特征、身份关联、逐帧一致率）、图优化、相机标定和地面纹理。可以只给需要的帧出标签（`frames.frame_stride`、`features.frame_stride`）、减小回环检测密度（`loops.stride`）、减少标定轨迹数（`calibration.tracks_per_anchor`）。逐帧标签和地面纹理的 C++ 调用都释放了 GIL，还可以按帧多线程并行（尚未做）。
- C++ 核心的文件读写目前由 Python 完成（trip 目录里的 npz / yaml / csv / jpg）；若要不带 Python 部署，需要补一个 C++ 数据读写层，算法接口不用改。
- 标线只处理与柱网对齐的直线（斜列车位需放宽 `markings.max_axis_angle`），箭头和文字尚未识别。
- 坡道只建模为平面；螺旋坡道需要分段平面或曲面。
- 位姿集合的剩余缺口：LiDAR–IMU 外参与时间偏差未估计，IMU 杆臂未建模。
- LiDAR BA 把位姿块当稠密矩阵求解（先消去惯性状态），适用于约 2500 个关键帧以内。更大的车库要走稀疏求解（已有，但填充多、较慢）或分层 BA（HBA 式子图）。
- 地图的绝对坐标只由入口 INS / GNSS 决定。测量级成果要用控制点（手持真值或全站仪靶标）定坐标系，见“测量级验收”。
- 手持真值目前是模拟的。真实项目需要把手持点云放成约定目录，并在其上数字化柱、车位、标线作为检查要素。完整度的中部高度带包含两次采集之间变化的停放车辆。
- 入口线被遮挡多的排只有很少的可靠端点（small 数据里有一排偏 4–5 cm）；斜列车位和带横线的车位（T / L 形）还没有专门处理。
- 语言路线：
  - 目前用合成真值实现验证。真实模型（SAM 2 / SAM 3、SigLIP 2、DINO / RoMa、Depth Anything 3、OCR）的适配器属于方案第 3 步，需要 GPU 服务器。
  - 相机看不到的面（车顶等）没有 2D 依据，需要点云网络补全（方案第 5 步）。
  - 逐帧标签由点云渲染得到，边界不如 2D 掩膜，3DGS 渲染（第 3 步）会改善。
- 新写和改动的 Python 模块已按 Google 风格编写；其余旧模块（markings、slots、structure、objects4d、levels、bev 等）尚未改写。
