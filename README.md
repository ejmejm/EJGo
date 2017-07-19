# EJGo

EJ Go Bot by Edan Meyer

## What it is

This is a bot I am developing to play the game Go. The bot uses a deep convolutional network and trains of the moves of skilled (6 dan+) players to choose its moves. It currently has an accuracy of about 50% in choosing the correct moves. In comparison, the state of the art Alpha Go bot holds an accuracy of about 57%.

## Install

### Prereqs

- [Tensorflow](https://www.tensorflow.org/install/)
- [Tflearn](https://github.com/tflearn/tflearn)

### Setup

1. To install EJGo, first clone the repository to your computer.
2. Download [sgfmill](https://github.com/mattheww/sgfmill) and put the sgfmill folder in the main directory of the repository
3. Download [sgf files](https://www.u-go.net/gamerecords/) for training (preferably at least 20,000+ files). Create a new folder called kifu in the main directory and place the sgf files in the kifu folder.
4. Download and install [GNUGo](https://www.gnu.org/software/gnugo/download.html), and make sure that the command `gnugo --mode gtp --level 10 --chineese-rules --positional-superko` works without error. On Windows this can be achieved through putting the executable and attached dlls in the repository main directory. On Ubuntu GNUGo can be installed with the terminal command `sudo apt-get install gnugo`.
5. (Optional) Download the [kgsgtp.jar](https://www.gokgs.com/download.jsp) file to be able to run the bot on the KGS Server. Add an "ioproperties.ini" file as per the specifications that come with the kgsgtp download.

## Models

## Usage

To train 

## Compatibility

EJGo was originally developed on Windows 10, so it should be fully compatible with Windows. It should be also mostly compatible with Linux, but it is a work in progress, so there are likely to be more errors. Please submit any errors as an issue so I can fix them up.

Thank you!

