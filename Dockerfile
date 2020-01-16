ARG BUILD_ARCH=amd64
FROM ${BUILD_ARCH}/debian:buster-slim

COPY pyinstaller/dist/* /usr/lib/rhasspyserver_hermes/
COPY debian/bin/* /usr/bin/
COPY web/ /usr/lib/rhasspyserver_hermes/web/

WORKDIR /usr/lib/rhasspyserver_hermes

ENTRYPOINT ["/usr/bin/rhasspy-server-hermes"]
