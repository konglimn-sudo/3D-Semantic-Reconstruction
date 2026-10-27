# gr_localize and gr_recon without Python: the vehicle-side localizer in a
# reconstructed map and the geometric stages of the offline reconstruction.
#
#   docker build -f docker/Dockerfile -t garage-recon .
#   docker run --rm -v "$PWD/work:/work" -v "$PWD/data:/data" garage-recon \
#     gr_localize --nav /work/semantics/nav --trip /data/trip_03 \
#                 --out /work/loc/trip_03
#
# Without --config, gr_localize reads the reconstruction's
# config.used.yaml; a new configuration starts from
# /opt/garage_recon/share/garage_recon/default_config.yaml.

FROM ubuntu:24.04 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      cmake g++ make libeigen3-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY cpp cpp
COPY python/garage_recon/default_config.yaml python/garage_recon/
RUN cmake -S . -B build -DGR_BUILD_PYTHON=OFF -DGR_INPLACE=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/garage_recon \
 && cmake --build build -j"$(nproc)" \
 && ./build/gr_tests \
 && cmake --install build

FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 zlib1g \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/garage_recon /opt/garage_recon
ENV PATH=/opt/garage_recon/bin:$PATH
WORKDIR /work
CMD ["gr_localize"]
