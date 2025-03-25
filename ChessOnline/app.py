from flask import Flask, render_template, request, jsonify, session
import uuid
import json
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)


# Helper function to validate a chess move
def is_valid_move(board, from_row, from_col, to_row, to_col, current_player):
    # Basic validation - would be expanded in a real implementation
    if from_row < 0 or from_row > 7 or from_col < 0 or from_col > 7:
        return False
    if to_row < 0 or to_row > 7 or to_col < 0 or to_col > 7:
        return False

    # Check if source position has a piece
    if board[from_row][from_col] is None:
        return False

    # Check if the piece belongs to the current player
    if board[from_row][from_col]['color'] != current_player:
        return False

    # Check if destination has a piece of the same color
    if board[to_row][to_col] is not None and board[to_row][to_col]['color'] == current_player:
        return False

    return True  # Further validation would be implemented in the frontend


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/start-game', methods=['POST'])
def start_game():
    data = request.json
    starting_player = data.get('starting_player', 'white')  # Default to white if not specified

    game_id = str(uuid.uuid4())
    session['game_id'] = game_id
    session['board'] = [[None for _ in range(8)] for _ in range(8)]
    session['current_player'] = starting_player  # Use the starting player from the coin flip

    return jsonify({
        'success': True,
        'game_id': game_id,
        'current_player': starting_player
    })


@app.route('/api/make-move', methods=['POST'])
def make_move():
    data = request.json

    game_id = data.get('game_id')
    from_row = data.get('from_row')
    from_col = data.get('from_col')
    to_row = data.get('to_row')
    to_col = data.get('to_col')

    # Validate game ID
    if 'game_id' not in session or session['game_id'] != game_id:
        return jsonify({'success': False, 'error': 'Invalid game ID'})

    # Validate move
    if not is_valid_move(session['board'], from_row, from_col, to_row, to_col, session['current_player']):
        return jsonify({'success': False, 'error': 'Invalid move'})

    # Update board state
    session['board'][to_row][to_col] = session['board'][from_row][from_col]
    session['board'][from_row][from_col] = None

    # Switch player
    session['current_player'] = 'black' if session['current_player'] == 'white' else 'white'

    return jsonify({
        'success': True,
        'current_player': session['current_player']
    })


@app.route('/api/check-status', methods=['POST'])
def check_status():
    data = request.json
    game_id = data.get('game_id')

    # Validate game ID
    if 'game_id' not in session or session['game_id'] != game_id:
        return jsonify({'success': False, 'error': 'Invalid game ID'})

    # Check for checkmate, stalemate, etc.
    status = {
        'current_player': session['current_player'],
        'in_check': False,  # Would be computed in a real implementation
        'checkmate': False,
        'stalemate': False
    }

    return jsonify({'success': True, 'status': status})


if __name__ == '__main__':
    app.run(debug=True)