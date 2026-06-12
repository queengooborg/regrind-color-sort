# Plastic Regrind Color Sort

> [!NOTE]
> If you enjoy this project and want to help with its maintenance, please consider supporting me via Ko-Fi!
>
> <a href='https://ko-fi.com/queengooborg' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi4.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

This is a project for automating the sorting process of plastic regrind by color with the intent of aiding the recycling of failed 3D prints and filament change purge into new filament. This is designed to run on a Raspberry Pi in order to drive the computer vision algorithm whilst also providing GPIO pins for the electronics.

Currently, only the CV portion has been completed. The hardware portion is still in the works.

## Requirements

- Raspberry Pi
  - Python 3.11+
  - I2C enabled in `raspi-config`
- Raspberry Pi camera
  - A webcam also works, but not recommended for the hardware setup
- A solid background (neutral gray preferred)
- A 3D printer
  - Recommended: 256mm x 256mm build plate
  - Minimum: 180mm x 180mm build plate
- More TBD

## Hardware Design

The following is a quick mockup of the intended hardware design for this. The design may change as development progresses.

![](./mockup.png)

The hardware will be designed for regrind created with P-4 (4x38mm) security level shredder or smaller.

## 3D Printing

> [!NOTE]
> This design is not yet finished. Please wait until models for the hopper, feeder, Pi holder, etc. have been created.

The 3D printed parts are designed to fit on the build plate of a Bambu Lab X1 Carbon -- in other words, 256mm x 256mm. However, by cutting larger models in half, they can safely fit on the build plate of an A1 mini (180mm x 180mm).

I recommend the following print settings:

- Material: Any Except TPU (PLA or PETG Recommended)
- Layer Height: 0.2
- Walls: 2
- Infill: 15%
- Support: No
- Raft/Brim: No\*
  - If your printer isn't printing the flaps well, enable the brim for those

You'll need to print all of the models in the `models/` folder, most needing to be printed multiple times. Here are the counts:

| Model          | Quantity |
| -------------- | -------- |
| Bin            | 17       |
| Bin Holder     | 17       |
| Flap           | 15       |
| Flap Connector | 8        |
| Stage 1 Paths  | 1        |
| Stage 2 Paths  | 2        |
| Stage 3 Paths  | 4        |
| Reject Flap    | 1        |
| Reject Path    | 1        |

## Assembly

As the design is still in development, a step-by-step guide is not yet made. For now, please review the CAD files to determine where each piece goes.

To attach the stages together, you can either use glue, such as B-6000 or superglue, to glue the pieces, or you can run a low-heat soldering iron along the seams and weld the pieces together.

## Setup

Run `install.sh` to install everything.

## How to Use

The hardware portion is still in the works, so currently, this is a proof-of-concept for the CV side. To use the CV side:

- Position a camera and a light facing down onto a solid, flat surface
  - Preferably, the background should be a uniform color that doesn't match any of your filament scraps
- Press the space bar to identify the background
- For every color of filament you have...
  - Place a filament scrap on the surface
  - Press the period key to define a new color and a quick-assign key
  - If a scrap is misrecognized as another color, either use the quick-assign key if the color is defined, or press period to define the new color
  - For best results, place one sample in frame at a time

## Disclaimer: Use of AI

As this is a hobby project and I am not all that familiar with computer vision or microcontrollers, unfortunately a majority of code was written by ChatGPT, and ChatGPT was used extensively to help figure out what parts were needed. I hate AI, and I hate that I'm using it so heavily in this project...but it is what it is.

I do hope that I will eventually be able to replace all the AI-generated code with code written by real people.
