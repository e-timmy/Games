class Enemy {
    constructor(x, y, size) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.speed = 1; // Slower than the player
        this.direction = { dx: 0, dy: 0 };
    }

    update(maze, player) {
        // Simple A* pathfinding
        let playerCol = Math.floor(player.x / CELL_SIZE);
        let playerRow = Math.floor(player.y / CELL_SIZE);
        let enemyCol = Math.floor(this.x / CELL_SIZE);
        let enemyRow = Math.floor(this.y / CELL_SIZE);

        // Determine direction
        if (playerCol > enemyCol && !maze.grid[enemyRow][enemyCol].walls.right) {
            this.direction.dx = 1;
            this.direction.dy = 0;
        } else if (playerCol < enemyCol && !maze.grid[enemyRow][enemyCol].walls.left) {
            this.direction.dx = -1;
            this.direction.dy = 0;
        } else if (playerRow > enemyRow && !maze.grid[enemyRow][enemyCol].walls.bottom) {
            this.direction.dx = 0;
            this.direction.dy = 1;
        } else if (playerRow < enemyRow && !maze.grid[enemyRow][enemyCol].walls.top) {
            this.direction.dx = 0;
            this.direction.dy = -1;
        }

        // Move
        this.x += this.direction.dx * this.speed;
        this.y += this.direction.dy * this.speed;

        // Ensure enemy stays within cell boundaries
        this.x = Math.max(0, Math.min(this.x, (maze.cols - 1) * CELL_SIZE));
        this.y = Math.max(0, Math.min(this.y, (maze.rows - 1) * CELL_SIZE));
    }

    draw(ctx) {
        ctx.fillStyle = '#f00';
        ctx.fillRect(this.x, this.y, this.size, this.size);
    }

    collidesWith(player) {
        return (
            this.x < player.x + player.size &&
            this.x + this.size > player.x &&
            this.y < player.y + player.size &&
            this.y + this.size > player.y
        );
    }
}