# Rhasspy Hermes Web Server


[![Continous Integration](https://github.com/rhasspy/rhasspy-server-hermes/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-server-hermes/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-server-hermes.svg)](https://github.com/rhasspy/rhasspy-server-hermes/blob/master/LICENSE)

Web interface to Rhasspy using Hermes protocol on the back end.

## Running With Docker

```bash
docker run -it rhasspy/rhasspy-server-hermes:<VERSION> <ARGS>
```

## Building From Source

Clone the repository and create the virtual environment:

```bash
git clone https://github.com/rhasspy/rhasspy-server-hermes.git
cd rhasspy-server-hermes
make venv
```

Before running, you will need to put the generated artifacts (`dist` directory) from [rhasspy-web-vue](https://github.com/rhasspy/rhasspy-web-vue) in a directory named `web`.

Run the `bin/rhasspy-server-hermes` script to access the command-line interface:

```bash
bin/rhasspy-server-hermes --help
```

## Building the Debian Package

Before building, you will need to put the generated artifacts (`dist` directory) from [rhasspy-web-vue](https://github.com/rhasspy/rhasspy-web-vue) in a directory named `web`.

Follow the instructions to build from source, then run:


```bash
source .venv/bin/activate
make debian
```

If successful, you'll find a `.deb` file in the `dist` directory that can be installed with `apt`.

## Building the Docker Image

Before building, you will need to put the generated artifacts (`dist` directory) from [rhasspy-web-vue](https://github.com/rhasspy/rhasspy-web-vue) in a directory named `web`.

Follow the instructions to build from source, then run:

```bash
source .venv/bin/activate
make docker
```

This will create a Docker image tagged `rhasspy/rhasspy-server-hermes:<VERSION>` where `VERSION` comes from the file of the same name in the source root directory.

NOTE: If you add things to the Docker image, make sure to whitelist them in `.dockerignore`.

## Command-Line Options

```
usage: Rhasspy [-h] --profile PROFILE [--host HOST] [--port PORT]
               [--mqtt-host MQTT_HOST] [--mqtt-port MQTT_PORT]
               [--local-mqtt-port LOCAL_MQTT_PORT]
               [--system-profiles SYSTEM_PROFILES]
               [--user-profiles USER_PROFILES] [--set SET SET] [--ssl SSL SSL]
               [--log-level LOG_LEVEL] [--web-dir WEB_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --profile PROFILE, -p PROFILE
                        Name of profile to load
  --host HOST           Host for web server
  --port PORT           Port for web server
  --mqtt-host MQTT_HOST
                        Host for MQTT broker
  --mqtt-port MQTT_PORT
                        Port for MQTT broker
  --local-mqtt-port LOCAL_MQTT_PORT
                        Port to use for internal MQTT broker (default: 12183)
  --system-profiles SYSTEM_PROFILES
                        Directory with base profile files (read only)
  --user-profiles USER_PROFILES
                        Directory with user profile files (read/write)
  --set SET SET, -s SET SET
                        Set a profile setting value
  --ssl SSL SSL         Use SSL with <CERT_FILE <KEY_FILE>
  --log-level LOG_LEVEL
                        Set logging level
  --web-dir WEB_DIR     Directory with compiled Vue site (default: dist)
```
