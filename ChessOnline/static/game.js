document.addEventListener('DOMContentLoaded', function() {
    const boardWrapper = document.getElementById('chess-board-wrapper');
    const board = document.getElementById('chess-board');
    const coinFlip = document.getElementById('coin-flip');
    const coin = document.getElementById('coin');
    const flipButton = document.getElementById('flip-button');
    const timer = document.getElementById('timer');
    const turnIndicator = document.getElementById('turn-indicator');
    const checkIndicator = document.getElementById('check-indicator');
    const resetButton = document.getElementById('reset-game');
    const gameStatus = document.getElementById('game-status');

    // Board coordinate labels
    const filesTop = document.querySelector('.files-top');
    const filesBottom = document.querySelector('.files-bottom');
    const ranksLeft = document.querySelector('.ranks-left');
    const ranksRight = document.querySelector('.ranks-right');

    // Game state
    let gameState = {
        board: Array(8).fill().map(() => Array(8).fill(null)),
        selectedPiece: null,
        selectedTile: null,
        currentPlayer: 'white',
        timeLeft: 60,
        timerInterval: null,
        gameActive: false,
        castlingRights: {
            white: { kingSide: true, queenSide: true },
            black: { kingSide: true, queenSide: true }
        },
        enPassantTarget: null,
        kings: { white: null, black: null },
        moveHistory: [],
        checkStatus: { white: false, black: false },
        gameId: null,
        startingPlayer: 'white', // Track the starting player from the coin flip
        boardOrientation: 'white', // Track which color is at the bottom of the board
        isRotating: false // Track if the board is currently rotating
    };

    // Create the board UI
    function createBoard() {
        board.innerHTML = '';
        createBoardCoordinates(); // Create board coordinates

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const tile = document.createElement('div');
                tile.classList.add('chess-tile');
                tile.classList.add((row + col) % 2 === 0 ? 'light' : 'dark');

                // Algebraic notation
                const notation = String.fromCharCode(97 + col) + (8 - row);
                tile.dataset.position = notation;
                tile.dataset.row = row;
                tile.dataset.col = col;

                tile.addEventListener('click', handleTileClick);
                board.appendChild(tile);
            }
        }
    }

    // Create board coordinates (a-h, 1-8)
    function createBoardCoordinates() {
        const files = 'abcdefgh';

        // Clear existing coordinates
        filesTop.innerHTML = '';
        filesBottom.innerHTML = '';
        ranksLeft.innerHTML = '';
        ranksRight.innerHTML = '';

        // Create file labels (a-h)
        for (let i = 0; i < 8; i++) {
            const topLabel = document.createElement('div');
            topLabel.classList.add('file-label');
            topLabel.textContent = files[i];
            filesTop.appendChild(topLabel);

            const bottomLabel = document.createElement('div');
            bottomLabel.classList.add('file-label');
            bottomLabel.textContent = files[i];
            filesBottom.appendChild(bottomLabel);
        }

        // Create rank labels (1-8)
        for (let i = 0; i < 8; i++) {
            const leftLabel = document.createElement('div');
            leftLabel.classList.add('rank-label');
            leftLabel.textContent = 8 - i;
            ranksLeft.appendChild(leftLabel);

            const rightLabel = document.createElement('div');
            rightLabel.classList.add('rank-label');
            rightLabel.textContent = 8 - i;
            ranksRight.appendChild(rightLabel);
        }
    }

    // Initialize the chess pieces
    function initializeGame() {
        // Start a new game on the server
        fetch('/api/start-game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                starting_player: gameState.startingPlayer // Send the starting player from coin flip
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                gameState.gameId = data.game_id;
                gameState.currentPlayer = gameState.startingPlayer; // Use our coin flip result

                // Reset local game state
                gameState.board = Array(8).fill().map(() => Array(8).fill(null));
                gameState.selectedPiece = null;
                gameState.selectedTile = null;
                gameState.timeLeft = 60;
                gameState.gameActive = true;
                gameState.castlingRights = {
                    white: { kingSide: true, queenSide: true },
                    black: { kingSide: true, queenSide: true }
                };
                gameState.enPassantTarget = null;
                gameState.kings = { white: null, black: null };
                gameState.moveHistory = [];
                gameState.checkStatus = { white: false, black: false };

                // Set board orientation to match starting player
                gameState.boardOrientation = gameState.startingPlayer;

                // Show the board, hide coin flip
                boardWrapper.style.display = 'block';
                coinFlip.style.display = 'none';

                // Clear the board UI
                clearBoard();

                // Setup pieces
                setupPieces();

                // Update the UI
                updateBoardUI();
                turnIndicator.textContent = `${gameState.currentPlayer.charAt(0).toUpperCase() + gameState.currentPlayer.slice(1)}'s turn`;
                checkIndicator.textContent = '';
                gameStatus.textContent = '';

                // Start the timer
                resetTimer();
            } else {
                console.error('Failed to start a new game:', data.error);
            }
        })
        .catch(error => {
            console.error('Error starting game:', error);
        });
    }

    function clearBoard() {
        // Remove all pieces from the board
        const tiles = document.querySelectorAll('.chess-tile');
        tiles.forEach(tile => {
            tile.innerHTML = '';
            tile.classList.remove('highlight-blue', 'highlight-valid-move', 'highlight-check');
        });
    }

    function setupPieces() {
        // Setup pawns
        for (let col = 0; col < 8; col++) {
            placePiece(1, col, createPiece('pawn', 'black'));
            placePiece(6, col, createPiece('pawn', 'white'));
        }

        // Setup rooks
        placePiece(0, 0, createPiece('rook', 'black'));
        placePiece(0, 7, createPiece('rook', 'black'));
        placePiece(7, 0, createPiece('rook', 'white'));
        placePiece(7, 7, createPiece('rook', 'white'));

        // Setup knights
        placePiece(0, 1, createPiece('knight', 'black'));
        placePiece(0, 6, createPiece('knight', 'black'));
        placePiece(7, 1, createPiece('knight', 'white'));
        placePiece(7, 6, createPiece('knight', 'white'));

        // Setup bishops
        placePiece(0, 2, createPiece('bishop', 'black'));
        placePiece(0, 5, createPiece('bishop', 'black'));
        placePiece(7, 2, createPiece('bishop', 'white'));
        placePiece(7, 5, createPiece('bishop', 'white'));

        // Setup queens
        placePiece(0, 3, createPiece('queen', 'black'));
        placePiece(7, 3, createPiece('queen', 'white'));

        // Setup kings
        placePiece(0, 4, createPiece('king', 'black'));
        placePiece(7, 4, createPiece('king', 'white'));

        // Store king positions
        gameState.kings.black = { row: 0, col: 4 };
        gameState.kings.white = { row: 7, col: 4 };
    }

    function createPiece(type, color) {
        return {
            type,
            color,
            moved: false
        };
    }

    function placePiece(row, col, piece) {
        gameState.board[row][col] = piece;
    }

    function updateBoardUI() {
        const tiles = document.querySelectorAll('.chess-tile');

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const index = row * 8 + col;
                const piece = gameState.board[row][col];
                const tile = tiles[index];

                // Clear the tile
                tile.innerHTML = '';

                if (piece) {
                    const pieceElement = document.createElement('div');
                    pieceElement.classList.add('chess-piece');
                    pieceElement.classList.add(`${piece.color}-piece`);
                    pieceElement.textContent = getPieceSymbol(piece);
                    pieceElement.dataset.type = piece.type;
                    pieceElement.dataset.color = piece.color;
                    tile.appendChild(pieceElement);
                }
            }
        }

        // Highlight king in check
        if (gameState.checkStatus.white) {
            const kingRow = gameState.kings.white.row;
            const kingCol = gameState.kings.white.col;
            const kingTileIndex = kingRow * 8 + kingCol;
            tiles[kingTileIndex].classList.add('highlight-check');
        } else {
            tiles.forEach(tile => {
                if (tile.classList.contains('highlight-check')) {
                    tile.classList.remove('highlight-check');
                }
            });
        }

        if (gameState.checkStatus.black) {
            const kingRow = gameState.kings.black.row;
            const kingCol = gameState.kings.black.col;
            const kingTileIndex = kingRow * 8 + kingCol;
            tiles[kingTileIndex].classList.add('highlight-check');
        }
    }

    function getPieceSymbol(piece) {
        const symbols = {
            'king': { 'white': '♔', 'black': '♚' },
            'queen': { 'white': '♕', 'black': '♛' },
            'rook': { 'white': '♖', 'black': '♜' },
            'bishop': { 'white': '♗', 'black': '♝' },
            'knight': { 'white': '♘', 'black': '♞' },
            'pawn': { 'white': '♙', 'black': '♟' }
        };
        return symbols[piece.type][piece.color];
    }

    function handleTileClick(event) {
        if (!gameState.gameActive || gameState.isRotating) return;

        // Clear all highlights first
        clearHighlights();

        // Get the clicked tile and its position
        const tile = event.target.closest('.chess-tile');
        if (!tile) return;

        // Parse row and col in a way that accounts for board orientation
        const row = parseInt(tile.dataset.row);
        const col = parseInt(tile.dataset.col);

        // For debugging
        console.log(`Clicked tile at row ${row}, col ${col}`);
        console.log(`Current player: ${gameState.currentPlayer}`);

        // If a piece is already selected
        if (gameState.selectedPiece) {
            const selectedRow = parseInt(gameState.selectedTile.dataset.row);
            const selectedCol = parseInt(gameState.selectedTile.dataset.col);

            console.log(`Selected piece at row ${selectedRow}, col ${selectedCol}`);

            // If clicking the same piece, deselect it
            if (selectedRow === row && selectedCol === col) {
                gameState.selectedPiece = null;
                gameState.selectedTile = null;
                return;
            }

            // If clicking another piece of the same color, select that piece instead
            const clickedPiece = gameState.board[row][col];
            if (clickedPiece && clickedPiece.color === gameState.currentPlayer) {
                selectPiece(tile, row, col);
                return;
            }

            // Otherwise, try to move the selected piece
            const isValid = isValidMove(selectedRow, selectedCol, row, col);
            console.log(`Is valid move: ${isValid}`);

            if (isValid) {
                console.log(`Moving piece from (${selectedRow},${selectedCol}) to (${row},${col})`);
                // Execute the move directly without server validation for now
                movePiece(selectedRow, selectedCol, row, col);
                gameState.selectedPiece = null;
                gameState.selectedTile = null;

                // Switch player and rotate board
                switchPlayer();
            } else {
                // Invalid move
                tile.classList.add('flash-red');
                setTimeout(() => tile.classList.remove('flash-red'), 500);

                // Keep the current piece selected
                gameState.selectedTile.classList.add('highlight-blue');
                highlightValidMoves(selectedRow, selectedCol);
            }
        } else {
            // If no piece is selected, try to select a piece
            const piece = gameState.board[row][col];
            if (piece && piece.color === gameState.currentPlayer) {
                selectPiece(tile, row, col);
            }
        }
    }

    function selectPiece(tile, row, col) {
        console.log(`Selecting piece at row ${row}, col ${col}`);
        // Select this piece
        gameState.selectedPiece = gameState.board[row][col];
        gameState.selectedTile = tile;

        // Highlight the selected tile
        tile.classList.add('highlight-blue');

        // Show valid moves
        highlightValidMoves(row, col);
    }

    function clearHighlights() {
        document.querySelectorAll('.highlight-blue').forEach(t => {
            t.classList.remove('highlight-blue');
        });

        document.querySelectorAll('.highlight-valid-move').forEach(t => {
            t.classList.remove('highlight-valid-move');
        });
    }

    function highlightValidMoves(row, col) {
        for (let targetRow = 0; targetRow < 8; targetRow++) {
            for (let targetCol = 0; targetCol < 8; targetCol++) {
                if (isValidMove(row, col, targetRow, targetCol)) {
                    const targetTile = getTileFromPosition(targetRow, targetCol);
                    targetTile.classList.add('highlight-valid-move');
                }
            }
        }
    }

    function getTileFromPosition(row, col) {
        return document.querySelector(`.chess-tile[data-row="${row}"][data-col="${col}"]`);
    }

    function isValidMove(fromRow, fromCol, toRow, toCol) {
        // Can't move to the same position
        if (fromRow === toRow && fromCol === toCol) return false;

        const piece = gameState.board[fromRow][fromCol];
        if (!piece) return false;

        // Can't move opponent's pieces
        if (piece.color !== gameState.currentPlayer) return false;

        // Can't capture own pieces
        const targetPiece = gameState.board[toRow][toCol];
        if (targetPiece && targetPiece.color === piece.color) return false;

        let validMove = false;

        // Check piece-specific movement rules
        switch (piece.type) {
            case 'pawn':
                validMove = isValidPawnMove(fromRow, fromCol, toRow, toCol);
                break;
            case 'rook':
                validMove = isValidRookMove(fromRow, fromCol, toRow, toCol);
                break;
            case 'knight':
                validMove = isValidKnightMove(fromRow, fromCol, toRow, toCol);
                break;
            case 'bishop':
                validMove = isValidBishopMove(fromRow, fromCol, toRow, toCol);
                break;
            case 'queen':
                validMove = isValidQueenMove(fromRow, fromCol, toRow, toCol);
                break;
            case 'king':
                validMove = isValidKingMove(fromRow, fromCol, toRow, toCol);
                break;
        }

        // Check if the move would put/leave the king in check
        if (validMove) {
            // Make a temporary move and check if king is in check
            const originalBoard = JSON.parse(JSON.stringify(gameState.board));
            const originalKings = JSON.parse(JSON.stringify(gameState.kings));

            // Update king position if moving the king
            if (piece.type === 'king') {
                gameState.kings[piece.color].row = toRow;
                gameState.kings[piece.color].col = toCol;
            }

            // Make the move temporarily
            gameState.board[toRow][toCol] = gameState.board[fromRow][fromCol];
            gameState.board[fromRow][fromCol] = null;

            // Check if king is in check after the move
            const kingInCheck = isKingInCheck(piece.color);

            // Restore the original board and king positions
            gameState.board = originalBoard;
            gameState.kings = originalKings;

            // If the king would be in check, the move is invalid
            if (kingInCheck) {
                validMove = false;
            }
        }

        return validMove;
    }

    // All of the piece movement validation functions remain the same as before...
    function isValidPawnMove(fromRow, fromCol, toRow, toCol) {
        const piece = gameState.board[fromRow][fromCol];
        const direction = piece.color === 'white' ? -1 : 1;

        // Check for standard one-square forward move
        if (fromCol === toCol && toRow === fromRow + direction && !gameState.board[toRow][toCol]) {
            return true;
        }

        // Check for initial two-square forward move
        if (fromCol === toCol && !piece.moved &&
            toRow === fromRow + 2 * direction &&
            !gameState.board[fromRow + direction][fromCol] &&
            !gameState.board[toRow][toCol]) {
            return true;
        }

        // Check for diagonal capture
        if (Math.abs(fromCol - toCol) === 1 && toRow === fromRow + direction) {
            // Regular capture
            if (gameState.board[toRow][toCol] && gameState.board[toRow][toCol].color !== piece.color) {
                return true;
            }

            // En passant capture
            if (!gameState.board[toRow][toCol] && gameState.enPassantTarget &&
                gameState.enPassantTarget.row === toRow && gameState.enPassantTarget.col === toCol) {
                return true;
            }
        }

        return false;
    }

    function isValidRookMove(fromRow, fromCol, toRow, toCol) {
        // Rook moves horizontally or vertically
        if (fromRow !== toRow && fromCol !== toCol) return false;

        // Check if path is clear
        return isPathClear(fromRow, fromCol, toRow, toCol);
    }

    function isValidKnightMove(fromRow, fromCol, toRow, toCol) {
        // Knight moves in L-shape pattern
        return (Math.abs(fromRow - toRow) === 2 && Math.abs(fromCol - toCol) === 1) ||
               (Math.abs(fromRow - toRow) === 1 && Math.abs(fromCol - toCol) === 2);
    }

    function isValidBishopMove(fromRow, fromCol, toRow, toCol) {
        // Bishop moves diagonally
        if (Math.abs(fromRow - toRow) !== Math.abs(fromCol - toCol)) return false;

        // Check if path is clear
        return isPathClear(fromRow, fromCol, toRow, toCol);
    }

    function isValidQueenMove(fromRow, fromCol, toRow, toCol) {
        // Queen combines rook and bishop movements
        return isValidRookMove(fromRow, fromCol, toRow, toCol) ||
               isValidBishopMove(fromRow, fromCol, toRow, toCol);
    }

    function isValidKingMove(fromRow, fromCol, toRow, toCol) {
        // Standard king move (one square in any direction)
        if (Math.abs(fromRow - toRow) <= 1 && Math.abs(fromCol - toCol) <= 1) {
            return true;
        }

        // Check for castling
        if (fromRow === toRow && Math.abs(fromCol - toCol) === 2) {
            const piece = gameState.board[fromRow][fromCol];
            if (piece.moved) return false; // King must not have moved

            // Kingside castling
            if (toCol > fromCol) {
                // Check castling rights
                if (!gameState.castlingRights[piece.color].kingSide) return false;

                // Check if path is clear
                if (!isPathClear(fromRow, fromCol, fromRow, 7)) return false;

                // Check if rook is available
                const rook = gameState.board[fromRow][7];
                if (!rook || rook.type !== 'rook' || rook.color !== piece.color || rook.moved) {
                    return false;
                }

                // Check if king passes through check
                for (let col = fromCol; col <= fromCol + 2; col++) {
                    // Temporarily move the king
                    const tempKingPos = { row: fromRow, col: col };
                    const originalKingPos = { ...gameState.kings[piece.color] };
                    gameState.kings[piece.color] = tempKingPos;

                    // Check if king would be in check
                    if (isKingInCheck(piece.color)) {
                        // Reset king position
                        gameState.kings[piece.color] = originalKingPos;
                        return false;
                    }

                    // Reset king position
                    gameState.kings[piece.color] = originalKingPos;
                }

                return true;
            }
            // Queenside castling
            else {
                // Check castling rights
                if (!gameState.castlingRights[piece.color].queenSide) return false;

                // Check if path is clear
                if (!isPathClear(fromRow, fromCol, fromRow, 0)) return false;

                // Check if rook is available
                const rook = gameState.board[fromRow][0];
                if (!rook || rook.type !== 'rook' || rook.color !== piece.color || rook.moved) {
                    return false;
                }

                // Check if king passes through check
                for (let col = fromCol; col >= fromCol - 2; col--) {
                    // Temporarily move the king
                    const tempKingPos = { row: fromRow, col: col };
                    const originalKingPos = { ...gameState.kings[piece.color] };
                    gameState.kings[piece.color] = tempKingPos;

                    // Check if king would be in check
                    if (isKingInCheck(piece.color)) {
                        // Reset king position
                        gameState.kings[piece.color] = originalKingPos;
                        return false;
                    }

                    // Reset king position
                    gameState.kings[piece.color] = originalKingPos;
                }

                return true;
            }
        }

        return false;
    }

    function isPathClear(fromRow, fromCol, toRow, toCol) {
        const rowDirection = fromRow === toRow ? 0 : (toRow > fromRow ? 1 : -1);
        const colDirection = fromCol === toCol ? 0 : (toCol > fromCol ? 1 : -1);

        let currentRow = fromRow + rowDirection;
        let currentCol = fromCol + colDirection;

        while (currentRow !== toRow || currentCol !== toCol) {
            if (gameState.board[currentRow][currentCol]) {
                return false; // Path is blocked
            }
            currentRow += rowDirection;
            currentCol += colDirection;
        }

        return true;
    }

    function movePiece(fromRow, fromCol, toRow, toCol) {
        const piece = gameState.board[fromRow][fromCol];
        const targetPiece = gameState.board[toRow][toCol];

        // Store the move in history
        gameState.moveHistory.push({
            piece: { ...piece },
            from: { row: fromRow, col: fromCol },
            to: { row: toRow, col: toCol },
            captured: targetPiece ? { ...targetPiece } : null,
            enPassant: null,
            castling: null,
            promotion: null
        });

        // Handle en passant capture
        let capturedPawnRow = null;
        let capturedPawnCol = null;

        if (piece.type === 'pawn' && !targetPiece &&
            Math.abs(fromCol - toCol) === 1 &&
            gameState.enPassantTarget &&
            gameState.enPassantTarget.row === toRow &&
            gameState.enPassantTarget.col === toCol) {

            // Capture the pawn that moved two squares
            capturedPawnRow = piece.color === 'white' ? toRow + 1 : toRow - 1;
            capturedPawnCol = toCol;
            gameState.board[capturedPawnRow][capturedPawnCol] = null;

            // Update move history with en passant
            gameState.moveHistory[gameState.moveHistory.length - 1].enPassant = true;
            gameState.moveHistory[gameState.moveHistory.length - 1].captured = {
                ...gameState.board[capturedPawnRow][capturedPawnCol]
            };
        }

        // Handle castling
        if (piece.type === 'king' && Math.abs(fromCol - toCol) === 2) {
            // Kingside castling
            if (toCol > fromCol) {
                gameState.board[fromRow][fromCol + 1] = gameState.board[fromRow][7]; // Move rook
                gameState.board[fromRow][7] = null; // Remove rook from original position
                gameState.board[fromRow][fromCol + 1].moved = true; // Mark rook as moved

                // Update move history with castling
                gameState.moveHistory[gameState.moveHistory.length - 1].castling = 'kingside';
            }
            // Queenside castling
            else {
                gameState.board[fromRow][fromCol - 1] = gameState.board[fromRow][0]; // Move rook
                gameState.board[fromRow][0] = null; // Remove rook from original position
                gameState.board[fromRow][fromCol - 1].moved = true; // Mark rook as moved

                // Update move history with castling
                gameState.moveHistory[gameState.moveHistory.length - 1].castling = 'queenside';
            }
        }

        // Move the piece on the board
        gameState.board[toRow][toCol] = piece;
        gameState.board[fromRow][fromCol] = null;

        // Update king position if moving king
        if (piece.type === 'king') {
            gameState.kings[piece.color].row = toRow;
            gameState.kings[piece.color].col = toCol;
        }

        // Set en passant target if pawn moves two squares
        gameState.enPassantTarget = null;
        if (piece.type === 'pawn' && Math.abs(fromRow - toRow) === 2) {
            gameState.enPassantTarget = {
                row: (fromRow + toRow) / 2,
                col: fromCol
            };
        }

        // Update castling rights
        if (piece.type === 'king') {
            gameState.castlingRights[piece.color].kingSide = false;
            gameState.castlingRights[piece.color].queenSide = false;
        }
        else if (piece.type === 'rook') {
            if (fromCol === 0) { // Queenside rook
                gameState.castlingRights[piece.color].queenSide = false;
            }
            else if (fromCol === 7) { // Kingside rook
                gameState.castlingRights[piece.color].kingSide = false;
            }
        }

        // Mark the piece as moved
        piece.moved = true;

        // Handle pawn promotion (automatically promote to queen for simplicity)
        if (piece.type === 'pawn' && (toRow === 0 || toRow === 7)) {
            piece.type = 'queen';

            // Update move history with promotion
            gameState.moveHistory[gameState.moveHistory.length - 1].promotion = 'queen';
        }

        // Animate the piece movement on the UI
        animateMove(fromRow, fromCol, toRow, toCol, capturedPawnRow, capturedPawnCol);

        // Highlight the destination tile
        const toTile = getTileFromPosition(toRow, toCol);
        toTile.classList.add('highlight-blue');
        setTimeout(() => toTile.classList.remove('highlight-blue'), 800);
    }

    function animateMove(fromRow, fromCol, toRow, toCol, capturedPawnRow, capturedPawnCol) {
        const fromTile = getTileFromPosition(fromRow, fromCol);
        const toTile = getTileFromPosition(toRow, toCol);

        // If there was an en passant capture, animate the captured pawn
        if (capturedPawnRow !== null && capturedPawnCol !== null) {
            const capturedPawnTile = getTileFromPosition(capturedPawnRow, capturedPawnCol);
            const capturedPawn = capturedPawnTile.querySelector('.chess-piece');

            if (capturedPawn) {
                // Animate the pawn fading away
                capturedPawn.style.transition = 'opacity 0.3s ease';
                capturedPawn.style.opacity = '0';
                setTimeout(() => {
                    if (capturedPawnTile.contains(capturedPawn)) {
                        capturedPawnTile.removeChild(capturedPawn);
                    }
                }, 300);
            }
        }

        // Get the piece element and calculate its movement
        const pieceElement = fromTile.querySelector('.chess-piece');

        if (pieceElement) {
            // Calculate the distance to move
            const fromRect = fromTile.getBoundingClientRect();
            const toRect = toTile.getBoundingClientRect();
            const dx = toRect.left - fromRect.left;
            const dy = toRect.top - fromRect.top;

            // Apply the transformation
            pieceElement.style.transition = 'transform 0.5s ease';
            pieceElement.style.transform = `translate(${dx}px, ${dy}px)`;

            // Wait for the animation to finish, then update the board UI
            setTimeout(() => {
                updateBoardUI();

                // Check for check, checkmate, or stalemate
                checkGameState();
            }, 500);
        } else {
            updateBoardUI();
            checkGameState();
        }
    }

    // Rotate the board when switching players
    function rotateBoard() {
        if (gameState.isRotating) return;
        gameState.isRotating = true;

        // Determine if we need to rotate to white or black orientation
        const newOrientation = gameState.boardOrientation === 'white' ? 'black' : 'white';

        // Apply the appropriate animation class
        if (newOrientation === 'black') {
            boardWrapper.classList.remove('board-rotating-reverse');
            boardWrapper.classList.add('board-rotating');

            // Add CSS class for static rotation once animation completes
            setTimeout(() => {
                board.classList.add('chess-board-rotated');
                boardWrapper.classList.remove('board-rotating');
                gameState.boardOrientation = newOrientation;
                gameState.isRotating = false;
            }, 2000);
        } else {
            boardWrapper.classList.remove('board-rotating');
            boardWrapper.classList.add('board-rotating-reverse');

            // Remove CSS class for static rotation once animation completes
            setTimeout(() => {
                board.classList.remove('chess-board-rotated');
                boardWrapper.classList.remove('board-rotating-reverse');
                gameState.boardOrientation = newOrientation;
                gameState.isRotating = false;
            }, 2000);
        }
    }

    function switchPlayer() {
        gameState.currentPlayer = gameState.currentPlayer === 'white' ? 'black' : 'white';
        turnIndicator.textContent = `${gameState.currentPlayer.charAt(0).toUpperCase() + gameState.currentPlayer.slice(1)}'s turn`;

        // Rotate the board to match the new player
        rotateBoard();

        // Reset the timer only after the board rotation is complete
        setTimeout(() => {
            resetTimer();
        }, 2000);
    }

    function resetTimer() {
        clearInterval(gameState.timerInterval);
        gameState.timeLeft = 60;
        timer.textContent = gameState.timeLeft;
        startTimer();
    }

    function startTimer() {
        gameState.timerInterval = setInterval(() => {
            gameState.timeLeft--;
            timer.textContent = gameState.timeLeft;
            if (gameState.timeLeft === 0) {
                clearInterval(gameState.timerInterval);
                gameStatus.textContent = `Time's up! ${gameState.currentPlayer === 'white' ? 'Black' : 'White'} wins!`;
                gameState.gameActive = false;
            }
        }, 1000);
    }

    function isKingInCheck(color) {
        const kingRow = gameState.kings[color].row;
        const kingCol = gameState.kings[color].col;

        // Check if any opponent piece can capture the king
        const opponentColor = color === 'white' ? 'black' : 'white';

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = gameState.board[row][col];

                if (piece && piece.color === opponentColor) {
                    // For efficiency, skip checking if piece can reach the king
                    switch (piece.type) {
                        case 'pawn':
                            if (isValidPawnAttack(row, col, kingRow, kingCol, opponentColor)) {
                                return true;
                            }
                            break;
                        case 'rook':
                            if (isValidRookMove(row, col, kingRow, kingCol)) {
                                return true;
                            }
                            break;
                        case 'knight':
                            if (isValidKnightMove(row, col, kingRow, kingCol)) {
                                return true;
                            }
                            break;
                        case 'bishop':
                            if (isValidBishopMove(row, col, kingRow, kingCol)) {
                                return true;
                            }
                            break;
                        case 'queen':
                            if (isValidQueenMove(row, col, kingRow, kingCol)) {
                                return true;
                            }
                            break;
                        case 'king':
                            // King can only check another king if they're adjacent
                            if (Math.abs(row - kingRow) <= 1 && Math.abs(col - kingCol) <= 1) {
                                return true;
                            }
                            break;
                    }
                }
            }
        }

        return false;
    }

    function isValidPawnAttack(fromRow, fromCol, toRow, toCol, pawnColor) {
        const direction = pawnColor === 'white' ? -1 : 1;
        return Math.abs(fromCol - toCol) === 1 && toRow === fromRow + direction;
    }

    function hasLegalMoves(color) {
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = gameState.board[row][col];

                if (piece && piece.color === color) {
                    // Check if this piece has any legal moves
                    for (let targetRow = 0; targetRow < 8; targetRow++) {
                        for (let targetCol = 0; targetCol < 8; targetCol++) {
                            if (isValidMove(row, col, targetRow, targetCol)) {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        return false;
    }

    function checkGameState() {
        // Check if king is in check
        const whiteKingInCheck = isKingInCheck('white');
        const blackKingInCheck = isKingInCheck('black');

        // Update check status
        gameState.checkStatus.white = whiteKingInCheck;
        gameState.checkStatus.black = blackKingInCheck;

        // Update check indicator
        if (whiteKingInCheck) {
            checkIndicator.textContent = 'White King in Check!';
        } else if (blackKingInCheck) {
            checkIndicator.textContent = 'Black King in Check!';
        } else {
            checkIndicator.textContent = '';
        }

        // Check for checkmate or stalemate
        if (!hasLegalMoves(gameState.currentPlayer)) {
            if (gameState.checkStatus[gameState.currentPlayer]) {
                // Checkmate
                gameStatus.textContent = `Checkmate! ${gameState.currentPlayer === 'white' ? 'Black' : 'White'} wins!`;
            } else {
                // Stalemate
                gameStatus.textContent = 'Stalemate! The game is a draw.';
            }

            gameState.gameActive = false;
            clearInterval(gameState.timerInterval);
        }
    }

    // Event listeners
    flipButton.addEventListener('click', () => {
        coin.classList.add('flipping');

        // Hide the text while flipping
        coin.textContent = '';

        setTimeout(() => {
            // Truly random result
            const result = Math.random() < 0.5 ? 'White' : 'Black';

            // Set different appearance for each side of the coin
            if (result === 'White') {
                coin.style.background = 'linear-gradient(135deg, #fff, #aaa)';
                coin.style.color = '#000';
            } else {
                coin.style.background = 'linear-gradient(135deg, #555, #222)';
                coin.style.color = '#fff';
            }

            coin.textContent = result;

            // Store the starting player from coin flip
            gameState.startingPlayer = result.toLowerCase();
            gameState.currentPlayer = result.toLowerCase();
            gameState.boardOrientation = result.toLowerCase();

            console.log(`Coin flip result: ${result}`);

            // Start the game after showing the result
            setTimeout(() => {
                initializeGame();
            }, 1000);
        }, 2000);
    });

    resetButton.addEventListener('click', () => {
        clearInterval(gameState.timerInterval);

        // Reset visual state
        clearBoard();
        board.classList.remove('chess-board-rotated');
        boardWrapper.classList.remove('board-rotating', 'board-rotating-reverse');

        // Reset game state
        gameState.gameActive = false;
        gameState.selectedPiece = null;
        gameState.selectedTile = null;
        gameState.isRotating = false;
        gameState.boardOrientation = 'white';

        // Show coin flip, hide board
        coinFlip.style.display = 'block';
        boardWrapper.style.display = 'none';
        coin.textContent = '';
        coin.classList.remove('flipping');
        coin.style.background = 'linear-gradient(135deg, #0f0, #070)';

        // Clear game status
        turnIndicator.textContent = '';
        checkIndicator.textContent = '';
        gameStatus.textContent = '';
    });

    // Initialize the game board (but don't start a game yet)
    createBoard();

    // Hide the board initially, show the coin flip
    boardWrapper.style.display = 'none';
    coinFlip.style.display = 'block';
});