ARG BUILD_ARCH=amd64
FROM ${BUILD_ARCH}/python:3.7-alpine as build

RUN apk add --no-cache build-base

ENV VENV=/usr/.venv

RUN python3 -m venv $VENV
RUN $VENV/bin/pip3 install --upgrade pip

COPY requirements_rhasspy.txt requirements.txt /tmp/
RUN $VENV/bin/pip3 install -r /tmp/requirements_rhasspy.txt
RUN $VENV/bin/pip3 install -r /tmp/requirements.txt

# -----------------------------------------------------------------------------

ARG BUILD_ARCH=amd64
FROM ${BUILD_ARCH}/python:3.7-alpine

WORKDIR /rhasspy
COPY --from=build /usr/.venv /rhasspy/.venv/

COPY **/*.py /rhasspy/rhasspyserver_hermes/
COPY web/ /rhasspy/web/
COPY VERSION /rhasspy/

ENTRYPOINT ["/rhasspy/.venv/bin/python3", "-m", "rhasspyserver_hermes"]