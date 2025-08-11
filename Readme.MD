
# Running the Game + Server + DQN Agent

## 1. Clone the game repository
The game is located in a separate GitHub repository on your profile.  
Clone it locally:

```bash
git clone https://github.com/Ursusz/DumbHack-Survival
cd DumbHack-Survival
```

## 2. Create the Conda environment

Make sure you have conda installed.  
From the folder containing env.yaml:

```bash
conda env create -f env.yaml
```

Activate the environment:

```bash
conda activate tf26py38
```

## 3. Start the Flask server

The server handles HTTP communication between the game and the RL agent.

From the folder where you have server.py:

```bash
python server.py
```

The server will start on port 5002. Keep it running in a separate terminal.

## 4. Start the game

The game must be run in the same environment (`tf26py38`).

Open a **new terminal** and run the game executable from the GitHub release:

1. Download the latest release from the game's [repository](https://github.com/Ursusz/DumbHack-Survival) **Releases** section.  
2. Extract it to a folder of your choice.  
3. Run the executable (keep it open while the server and agent are running).

Example (Linux):

```bash
./game_executable
```

Example (Windows):

```powershell
.\game_executable.exe
```

## 5. Start the DQN agent

The agent uses the GameEnv class to communicate with the server via HTTP.

From the folder where you have agent.py:

```bash
conda activate tf26py38
python agent.py
```

## 6. Full execution flow

1. **Terminal 1**: Start the Flask server (server.py)
2. **Terminal 2**: Start the game
3. **Terminal 3**: Start the RL agent (agent.py)

All processes must use the same Conda environment: tf26py38.

## 7. Notes

- If the server or game fails to start, make sure port 5002 is free.

- If the agent stops with `requests.exceptions.ConnectionError`, check that both the game and server are running.

- You can tune hyperparameters in `agent.py` (e.g., `nb_steps`, `learning_rate`) for faster or slower training.

- Always keep the server running before starting the game or the agent.
