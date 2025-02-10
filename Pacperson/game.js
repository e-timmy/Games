const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');

const CELL_SIZE = 30;
const ROWS = 20;
const COLS = 20;
const NUM_ENEMIES = 3;
const NUM_POWERUPS = 5;

canvas.width = COLS * CELL_SIZE;
canvas.height = ROWS * CELL_SIZE;

const maze = new Maze(ROWS, COLS);
const player = new Player(0, 0, CELL_SIZE);
const enemies = [];
const powerUps = [];

// Initialize enemies at random positions
for (let i = 0; i < NUM_ENEMIES; i++) {
    // Place enemies away from the player's starting position
    let row, col;
    do {
        row = Math.floor(Math.random() * ROWS);
        col = Math.floor(Math.random() * COLS);
    } while (row < 3 && col < 3); // Avoid starting area

    enemies.push(new Enemy(
        col * CELL_SIZE,
        row * CELL_SIZE,
        CELL_SIZE
    ));
}

// Initialize power-ups at random positions
for (let i = 0; i < NUM_POWERUPS; i++) {
    let row, col;
    do {
        row = Math.floor(Math.random() * ROWS);
        col = Math.floor(Math.random() * COLS);
    } while (
        (row < 3 && col < 3) || // Avoid starting area
        (row === ROWS - 1 && col === COLS - 1) // Avoid end cell
    );

    powerUps.push(new PowerUp(
        col * CELL_SIZE,
        row * CELL_SIZE,
        CELL_SIZE
    ));
}

let gameCompleted = false;
let gameOver = false;

// Keep track of active keys
const activeKeys = new Set();

function gameLoop() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    maze.draw(ctx, CELL_SIZE);

    // Draw power-ups
    powerUps.forEach(powerUp => powerUp.draw(ctx));

    // Update and draw enemies
    enemies.forEach(enemy => {
        enemy.update(maze, player);
        enemy.draw(ctx);
    });

    player.update(maze, powerUps);
    player.draw(ctx);

    // Check win/lose conditions
    if (player.dead) {
        gameOver = true;
        ctx.fillStyle = 'white';
        ctx.font = '30px Arial';
        ctx.fillText('Game Over!', canvas.width / 2 - 70, canvas.height / 2);
    } else if (Math.floor(player.x / CELL_SIZE) === COLS - 1 &&
               Math.floor(player.y / CELL_SIZE) === ROWS - 1) {
        gameCompleted = true;
        ctx.fillStyle = 'white';
        ctx.font = '30px Arial';
        ctx.fillText('Level Complete!', canvas.width / 2 - 100, canvas.height / 2);
    }

    // Draw speed boost timer
    if (player.speedBoostTimer > 0) {
        ctx.fillStyle = 'white';
        ctx.font = '16px Arial';
        ctx.fillText(`Speed Boost: ${Math.ceil(player.speedBoostTimer / 60)}s`, 10, 20);
    }

    if (!gameCompleted && !gameOver) {
        requestAnimationFrame(gameLoop);
    }
}

document.addEventListener('keydown', (e) => {
    if (!gameCompleted && !gameOver && !activeKeys.has(e.key)) {
        activeKeys.add(e.key);

        switch (e.key) {
            case 'ArrowUp':
                player.setMoving(0, -1, true);
                break;
            case 'ArrowDown':
                player.setMoving(0, 1, true);
                break;
            case 'ArrowLeft':
                player.setMoving(-1, 0, true);
                break;
            case 'ArrowRight':
                player.setMoving(1, 0, true);
                break;
            case 'r':
                // Restart game
                location.reload();
                break;
        }
    }
});

document.addEventListener('keyup', (e) => {
    activeKeys.delete(e.key);

    switch (e.key) {
        case 'ArrowUp':
            if (player.currentDirection.dy === -1) player.setMoving(0, -1, false);
            break;
        case 'ArrowDown':
            if (player.currentDirection.dy === 1) player.setMoving(0, 1, false);
            break;
        case 'ArrowLeft':
            if (player.currentDirection.dx === -1) player.setMoving(-1, 0, false);
            break;
        case 'ArrowRight':
            if (player.currentDirection.dx === 1) player.setMoving(1, 0, false);
            break;
    }
});

gameLoop();