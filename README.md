# EJGo
### by Edan Meyer
## What it is

This is a bot I am developing to play the game Go. The bot uses a deep convolutional network and trains of the moves of skilled (6 dan+) players to choose its moves. It currently has an accuracy of about 52% in choosing the correct moves. In comparison, the state of the art Alpha Go bot holds an accuracy of about 57%.

## Install

### Prereqs

- [Tensorflow](https://www.tensorflow.org/install/)
- [Tflearn](https://github.com/tflearn/tflearn)

### Setup

1. To install EJGo, first clone the repository to your computer.
2. Download [sgfmill](https://github.com/mattheww/sgfmill) and put the sgfmill folder in the main directory of the repository
3. Download [sgf files](https://www.u-go.net/gamerecords/) for training (preferably at least 20,000+ files). Create a new folder called kifu in the main directory and place the sgf files in the kifu folder.
4. Download and install [GNUGo](https://www.gnu.org/software/gnugo/download.html), and make sure that the command `gnugo --mode gtp --level 10 --chineese-rules --positional-superko` works without error. On Windows this can be achieved through putting the executable and attached dlls in the repository main directory. On Ubuntu GNUGo can be installed with the terminal command `sudo apt-get install gnugo`.
5. (Optional) Download the [kgsgtp.jar](https://www.gokgs.com/download.jsp) file to be able to run the bot on the KGS Server. Add an `ioproperties.ini` file as per the specifications that come with the kgsgtp download.

## Models

All current network models are deep convolutional networks and can be found in the "models" folder. Each file is a seperate network model. The current best accuracy model is the ejmodel at about 50% accuracy. The network was based off the best network used [here](https://github.com/TheDuck314/go-NN). The network consists of 12 convolutional layers with elu activations and one fully connected layer followed with a softmax activation at the end. The first filter is 5x5, and the rest are 3x3. The model takes a 19x19x20 input array, 19x19 for the board size and 20 feature channels.

Note: Many models are outdated due to the fact that the input format for training changed several times during development

### Adding a New Model

To add a new model create a new python file in the models folder. The name of the file will be the name of the model itself. The name is restricted to letters only, no numbers or special characters. Every model file should contain a function called `get_network()` that contains and returns the network. I would recommend using `ejmodel.py` as a template.

## Usage

**Flags**
  - **n**: number of games to use
  - **m**: name of model to use
  - **s**: true to continue off a previous save or false to start with new weights(defaults to false)

**Commands**

* **Training**: `python train_network.py -n num_games -m model_name -s save`, eg: `python train_network.py -n 50000 -m ejmodel -s false`

* **Accuracy Testing**: `python test_accuracy.py -n num_games -m model_name`, eg: `python test_accuracy.py -n 100 -m ejmodel`

* **Host Bot on KGS Server**:
  * *Windows*: Run `StartServer.bat`
  * *Linux*: `sh StartServer.sh`
  
* **Play Local Game Against Bot**: `python play_game.py -m model_name`, eg: `python play_game.py -m ejmodel`

## File Purposes

- `go_nn.py` handles the forward propogation and network training
- `board_3d.py` contains functions to manipulate the board and build feature channels
- `board.py` is an old version of the board functions for 2D boards without feautre channels
- `global_vars_go.py` holds global variables when running the progarm
- `loader.py` takes care of loading models to tensorflow from the `models` folder
- `Engine.py` and `TFEngine.py` take care of basic engine functions for running games
- `GTP.py` communicates between the engine and uses Go to Text Protocol to make the bot easily compatible with most servers
- `KGSEngine.py` handles the engine specifically for the KGS Go Server
- `HelperEngine.py` communicates between the engine and GNU Go to know when to pass (passing is not handled by my bot)
- `board_ui.py` creates a Tkinter UI to play against the bot locally

## Compatibility

EJGo was originally developed on Windows 10, so it should be fully compatible with Windows. It should be also mostly compatible with Linux, but it is a work in progress, so there are likely to be more errors. Please submit any errors as an issue so I can fix them up.

## Acknowledgment

TheDuck314's [go-NN repository](https://github.com/TheDuck314/go-NN) was tremendously helpful. I used their `Engine.py`, `GTP.py`, `TFEngine.py`, `KGSEngine.py`, and `TFEngine.py` to handle nearly everything server communication related.

Thank you!

