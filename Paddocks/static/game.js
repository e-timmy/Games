class Game {
    constructor(player1Graphic, player2Graphic, gridSize) {
        this.player1 = {
            graphic: player1Graphic,
            score: 0
        };
        this.player2 = {
            graphic: player2Graphic,
            score: 0
        };
        this.gridSize = gridSize;
        this.currentPlayer = 1;
        this.board = [];

        this.initBoard();

        // Wait for color extraction before rendering
        Promise.all([
            this.getDominantColor(player1Graphic).then(color => {
                this.player1Color = color;
                const styleSheet = document.styleSheets[0];
                styleSheet.insertRule(`.line.player1 { background-color: ${color} !important; }`, styleSheet.cssRules.length);
            }),
            this.getDominantColor(player2Graphic).then(color => {
                this.player2Color = color;
                const styleSheet = document.styleSheets[0];
                styleSheet.insertRule(`.line.player2 { background-color: ${color} !important; }`, styleSheet.cssRules.length);
            })
        ]).then(() => {
            this.renderBoard();
            this.updateTurnDisplay();
        });
    }

    getDominantColor(imageSrc) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.src = imageSrc;
            img.onload = function() {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;
                let r = 0, g = 0, b = 0;

                for (let i = 0; i < data.length; i += 4) {
                    r += data[i];
                    g += data[i+1];
                    b += data[i+2];
                }

                r = Math.floor(r / (data.length / 4));
                g = Math.floor(g / (data.length / 4));
                b = Math.floor(b / (data.length / 4));

                resolve(`rgb(${r},${g},${b})`);
            };
        });
    }

    initBoard() {
        for (let i = 0; i <= this.gridSize; i++) {
            this.board[i] = [];
            for (let j = 0; j <= this.gridSize; j++) {
                this.board[i][j] = {
                    right: false,
                    bottom: false,
                    owner: null
                };
            }
        }
    }

    renderBoard() {
        const gameBoard = document.getElementById('game-board');
        gameBoard.innerHTML = '';

        const maxBoardSize = Math.min(window.innerWidth - 40, window.innerHeight - 200);
        const cellSize = maxBoardSize / this.gridSize;

        gameBoard.style.width = `${cellSize * this.gridSize}px`;
        gameBoard.style.height = `${cellSize * this.gridSize}px`;

        const gridContainer = document.createElement('div');
        gridContainer.style.position = 'absolute';
        gridContainer.style.width = '100%';
        gridContainer.style.height = '100%';
        gameBoard.appendChild(gridContainer);

        // Render lines and nodes
        for (let i = 0; i <= this.gridSize; i++) {
            for (let j = 0; j <= this.gridSize; j++) {
                // Nodes
                const node = document.createElement('div');
                node.classList.add('node');
                node.style.left = `${j * cellSize}px`;
                node.style.top = `${i * cellSize}px`;
                gridContainer.appendChild(node);

                // Horizontal lines
                if (j < this.gridSize) {
                    const hLine = document.createElement('div');
                    hLine.classList.add('line', 'horizontal-line');
                    if (this.board[i][j].right) {
                        hLine.classList.add(`player${this.board[i][j].rightPlayer}`);
                    }
                    hLine.dataset.row = i;
                    hLine.dataset.col = j;
                    hLine.dataset.type = 'h';
                    hLine.style.left = `${j * cellSize + 5}px`;
                    hLine.style.top = `${i * cellSize - 3}px`;
                    hLine.style.width = `${cellSize - 10}px`;
                    hLine.addEventListener('click', () => this.handleLineClick(i, j, 'h'));
                    gridContainer.appendChild(hLine);
                }

                // Vertical lines
                if (i < this.gridSize) {
                    const vLine = document.createElement('div');
                    vLine.classList.add('line', 'vertical-line');
                    if (this.board[i][j].bottom) {
                        vLine.classList.add(`player${this.board[i][j].bottomPlayer}`);
                    }
                    vLine.dataset.row = i;
                    vLine.dataset.col = j;
                    vLine.dataset.type = 'v';
                    vLine.style.left = `${j * cellSize - 3}px`;
                    vLine.style.top = `${i * cellSize + 5}px`;
                    vLine.style.height = `${cellSize - 10}px`;
                    vLine.addEventListener('click', () => this.handleLineClick(i, j, 'v'));
                    gridContainer.appendChild(vLine);
                }
            }
        }

        // Render completed squares
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (this.board[i][j].owner) {
                    const square = document.createElement('div');
                    square.classList.add('square');
                    square.style.left = `${j * cellSize}px`;
                    square.style.top = `${i * cellSize}px`;
                    square.style.width = `${cellSize}px`;
                    square.style.height = `${cellSize}px`;
                    square.style.backgroundImage = `url(${this.board[i][j].owner === 1 ? 
                        this.player1.graphic : this.player2.graphic})`;
                    gridContainer.appendChild(square);
                }
            }
        }
    }

    handleLineClick(row, col, type) {
        console.log(`Line clicked: row ${row}, col ${col}, type ${type}`);

        // Prevent re-clicking the same line
        if ((type === 'h' && this.board[row][col].right) ||
            (type === 'v' && this.board[row][col].bottom)) {
            console.log('Line already drawn');
            return;
        }

        // Mark the line as drawn and set the player
        if (type === 'h') {
            this.board[row][col].right = true;
            this.board[row][col].rightPlayer = this.currentPlayer;
        } else {
            this.board[row][col].bottom = true;
            this.board[row][col].bottomPlayer = this.currentPlayer;
        }

        console.log(`Line drawn by player ${this.currentPlayer}`);

        let scoredThisTurn = false;

        // Check all squares
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (!this.board[i][j].owner && this.isSquareComplete(i, j)) {
                    this.board[i][j].owner = this.currentPlayer;
                    scoredThisTurn = true;
                    if (this.currentPlayer === 1) {
                        this.player1.score++;
                    } else {
                        this.player2.score++;
                    }
                    console.log(`Square completed at (${i}, ${j}) by player ${this.currentPlayer}`);
                }
            }
        }

        this.updateScoreDisplay();
        this.renderBoard();  // Re-render to show changes

        if (!scoredThisTurn) {
            this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
        }
        this.updateTurnDisplay();

        console.log(`Current player after move: ${this.currentPlayer}`);
        console.log(`Player 1 score: ${this.player1.score}, Player 2 score: ${this.player2.score}`);

        if (this.isGameOver()) {
            this.endGame();
        }
    }

    // Ensure isSquareComplete method is correct
    isSquareComplete(row, col) {
        return (
            (this.board[row][col].right || false) &&
            (this.board[row][col+1] && this.board[row][col+1].bottom || false) &&
            (this.board[row+1] && this.board[row+1][col].right || false) &&
            (this.board[row][col].bottom || false)
        );
    }


    handleNodeClick(row, col) {
        if (this.firstNode === null) {
            this.firstNode = { row, col };
        } else {
            const dr = Math.abs(row - this.firstNode.row);
            const dc = Math.abs(col - this.firstNode.col);

            if ((dr === 0 && dc === 1) || (dr === 1 && dc === 0)) {
                this.drawLine(this.firstNode.row, this.firstNode.col, row, col);
                this.firstNode = null;
            } else {
                this.firstNode = { row, col };
            }
        }
    }

    drawLine(r1, c1, r2, c2) {
        const minRow = Math.min(r1, r2);
        const minCol = Math.min(c1, c2);

        if (r1 === r2) {  // Horizontal line
            if (this.board[r1][minCol].right) return;  // Line already exists
            this.board[r1][minCol].right = true;
        } else {  // Vertical line
            if (this.board[minRow][c1].bottom) return;  // Line already exists
            this.board[minRow][c1].bottom = true;
        }

        let scoredThisTurn = false;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (!this.board[i][j].owner && this.isSquareComplete(i, j)) {
                    this.board[i][j].owner = this.currentPlayer;
                    scoredThisTurn = true;
                    if (this.currentPlayer === 1) {
                        this.player1.score++;
                    } else {
                        this.player2.score++;
                    }
                }
            }
        }

        this.updateScoreDisplay();
        this.renderBoard();

        if (!scoredThisTurn) {
            this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
        }
        this.updateTurnDisplay();

        if (this.isGameOver()) {
            this.endGame();
        }
    }

    updateScoreDisplay() {
        document.getElementById('player1-score').textContent = this.player1.score;
        document.getElementById('player2-score').textContent = this.player2.score;
    }

    updateTurnDisplay() {
        const turnDisplay = document.getElementById('current-turn');
        turnDisplay.textContent = `Player ${this.currentPlayer}'s Turn`;
    }

    isGameOver() {
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                if (!this.board[i][j].right || !this.board[i][j].bottom) {
                    return false;
                }
            }
        }
        return true;
    }

    endGame() {
        const winner = this.player1.score > this.player2.score ? 1 :
                       this.player2.score > this.player1.score ? 2 : 0;
        alert(winner ? `Player ${winner} wins!` : "It's a tie!");
    }
}

document.addEventListener('DOMContentLoaded', () => {
    let player1Graphic, player2Graphic;

    document.querySelectorAll('.player-setup .graphic-selection img').forEach(img => {
        img.addEventListener('click', (e) => {
            const playerContainer = e.target.closest('.player-setup');
            playerContainer.querySelectorAll('img').forEach(i => i.classList.remove('selected'));
            e.target.classList.add('selected');

            if (playerContainer.querySelector('h2').textContent === 'Player 1') {
                player1Graphic = e.target.src;
            } else {
                player2Graphic = e.target.src;
            }
        });
    });

    document.getElementById('start-game').addEventListener('click', () => {
        if (!player1Graphic || !player2Graphic) {
            alert('Please select graphics for both players');
            return;
        }

        const gridSize = parseInt(document.getElementById('grid-size').value);
        document.getElementById('setup').classList.add('hidden');
        document.getElementById('game').classList.remove('hidden');

        document.getElementById('player1-graphic').src = player1Graphic;
        document.getElementById('player2-graphic').src = player2Graphic;

        new Game(player1Graphic, player2Graphic, gridSize);
    });
});