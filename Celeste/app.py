from flask import Flask, render_template, jsonify, request
from game_logic import GameState

app = Flask(__name__)
game_states = {}  # Store game states for different sessions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/game-state', methods=['GET'])
def get_game_state():
    session_id = request.args.get('session_id')
    if session_id not in game_states:
        game_states[session_id] = GameState()
    return jsonify(game_states[session_id].get_state())


@app.route('/api/update', methods=['POST'])
def update_game():
    data = request.json
    session_id = data.get('session_id')
    if session_id not in game_states:
        game_states[session_id] = GameState()

    game_state = game_states[session_id]
    input_data = data.get('input', {})
    print(f"Received input data: {input_data}")  # Debug log
    game_state.update(input_data)
    updated_state = game_state.get_state()
    print(f"Updated game state: {updated_state}")  # Debug log
    return jsonify(updated_state)


@app.route('/api/reset', methods=['POST'])
def reset_game():
    session_id = request.json.get('session_id')
    if session_id in game_states:
        game_states[session_id] = GameState()
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True)
