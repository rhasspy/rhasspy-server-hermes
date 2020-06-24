# Rhasspy Hermes Web Server


[![Continous Integration](https://github.com/rhasspy/rhasspy-server-hermes/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-server-hermes/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-server-hermes.svg)](https://github.com/rhasspy/rhasspy-server-hermes/blob/master/LICENSE)

Web interface to Rhasspy using Hermes protocol on the back end.

## Requirements

* Python 3.7
* [Quart](https://gitlab.com/pgjones/quart)
* [Hypercorn](https://pgjones.gitlab.io/hypercorn)
* [alpinejs](https://github.com/alpinejs/alpine/)

## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-server-hermes
$ cd rhasspy-server-hermes
$ ./configure
$ make
$ make install
```

## Running

```bash
$ bin/rhasspy-server-hermes <ARGS>
```

## Command-Line Options

```
usage: Rhasspy [-h] --profile PROFILE [--host HOST] [--port PORT]
               [--mqtt-host MQTT_HOST] [--mqtt-port MQTT_PORT]
               [--mqtt-username MQTT_USERNAME] [--mqtt-password MQTT_PASSWORD]
               [--local-mqtt-port LOCAL_MQTT_PORT]
               [--system-profiles SYSTEM_PROFILES]
               [--user-profiles USER_PROFILES] [--set SET SET]
               [--certfile CERTFILE] [--keyfile KEYFILE]
               [--log-level LOG_LEVEL] [--log-format LOG_FORMAT]
               [--web-dir WEB_DIR]

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
  --mqtt-username MQTT_USERNAME
                        Host for MQTT broker
  --mqtt-password MQTT_PASSWORD
                        Port for MQTT broker
  --local-mqtt-port LOCAL_MQTT_PORT
                        Port to use for internal MQTT broker (default: 12183)
  --system-profiles SYSTEM_PROFILES
                        Directory with base profile files (read only)
  --user-profiles USER_PROFILES
                        Directory with user profile files (read/write)
  --set SET SET, -s SET SET
                        Set a profile setting value
  --certfile CERTFILE   SSL certificate file
  --keyfile KEYFILE     SSL private key file (optional)
  --log-level LOG_LEVEL
                        Set logging level
  --log-format LOG_FORMAT
                        Python logger format
  --web-dir WEB_DIR     Directory with image/css/javascript files (default:
                        web)
```
