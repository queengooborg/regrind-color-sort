# Plastic Regrind Color Sort

This is a project for automating the sorting process of plastic regrind by color with the intent of aiding the recycling of failed 3D prints and filament change purge into new filament.  This is designed to run on a Raspberry Pi in order to drive the computer vision algorithm whilst also providing GPIO pins for the electronics.

Currently, only the CV portion has been completed.  The hardware portion is still in the works.

## Requirements

- Raspberry Pi
  - Python 3.11+
- A webcam or better
- A solid background (neutral gray preferred)
- A 3D printer (eventually)
- More TBD

## How to Use

The hardware portion is still in the works, so currently, this is a proof-of-concept for the CV side.  To use the CV side:

- Position a camera and a light facing down onto a solid, flat surface
  - Preferably, this should be a uniform color that doesn't match any of your filament scraps
- Press the space bar to identify the background
- For every color of filament you have...
	- Place a filament scrap on the surface
	- Press the period key to define a new color and a quick-assign key
	- If a scrap is misrecognized as another color, either use the quick-assign key if the color is defined, or press period to define the new color
	- For best results, place one sample in frame at a time

## Authoring Disclaimer

As this is a hobby project and I am not all that familiar with computer vision, let alone OpenCV, sadly the majority of code is written by ChatGPT.  I don't like the idea of using AI, let alone using it so heavily to write project code, but it did speed up the process to where I had a working version in a couple of hours, versus what might have been a couple of days or weeks.