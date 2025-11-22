# avionmesh

A python library to for interacting with Avi-on based lights

Forked from the original <https://github.com/oyvindkinsey/avionmqtt> repository, which will contain only the mqtt bits moving forward.

## support

This should support any devices that uses Avi-on's technology, including Halo Home and GE branded BLE lights (both discontinued, but both supported by Avi-on's cloud infra and mobile apps).

## features

- connects to the mesh using BLE
- support getting and setting color temperature (kelvin)
  - color temperature can be set *without* turning on the light
- supports getting and setting brightness
- supports managing both individual lights, groups, as well as the entire mesh at once
- supports sending date/time updates to the mesh
- supports polling for the state of the mesh

## acknowledgements

This project would not have been possible without the original work done in <https://github.com/nkaminski/csrmesh> and <https://github.com/nayaverdier/halohome>

## license

This project is licensed under the GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later). See the `LICENSE` file for the full license text and details.
