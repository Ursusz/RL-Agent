from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# werkzeug_logger = logging.getLogger('werkzeug')
# werkzeug_logger.disabled = True

GameState = None
Action = None

@app.route('/update_state', methods=['POST'])
def update_state_handler():
    global GameState
    
    # print(request.json)

    GameState = request.json
    if GameState:
        # print(data)
        return jsonify({"message": "Game state received"}), 200
    return jsonify({"message": "Invalid JSON"}), 400

@app.route('/get_state', methods=['GET'])
def get_state_handler():
    return jsonify(GameState)

@app.route('/send_action', methods=['POST'])
def send_action_handler():
    global Action

    action_data = request.json
    if action_data and 'action' in action_data:
        Action = action_data['action']
        return jsonify({"message": "Action received succesfully"}), 200
    return jsonify({"message": "Invalid JSON or missing action"}), 400

@app.route('/get_action', methods=['GET'])
def get_action_handler():
    global Action

    action_to_send = Action
    Action = None

    # print(f"actiune: {action_to_send}")
    if action_to_send is None:
        return jsonify({"action" : -1})
    return jsonify({"action": action_to_send})

@app.route('/reset_game', methods=['POST'])
def reset_game_handler():
    global Action

    Action = 4
    return jsonify({"message" : "success"}), 200

app.run(port=5002, debug=False)