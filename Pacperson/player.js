class PowerUp {
    constructor(x, y, size) {
        this.x = x;
        this.y = y;
        this.size = size;
    }

    draw(ctx) {
        ctx.fillStyle = '#00FFFF';
        ctx.beginPath();
        ctx.arc(
            this.x + this.size / 2,
            this.y + this.size / 2,
            this.size / 4,
            0,
            Math.PI * 2
        );
        ctx.fill();
    }
}

class Player {
    constructor(x, y, size) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.targetX = x;
        this.targetY = y;
        this.baseSpeed = 3;
        this.speed = this.baseSpeed;
        this.moving = false;
        this.currentDirection = { dx: 0, dy: 0 };
        this.dead = false;
        this.speedBoostTimer = 0;
    }

    update(maze, powerUps) {
        if (this.dead) return;

        if (this.moving && !this.isMoving()) {
            this.tryMove(this.currentDirection.dx, this.currentDirection.dy, maze);
        }

        let dx = this.targetX - this.x;
        let dy = this.targetY - this.y;
        let distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > this.speed) {
            let newX = this.x + (dx / distance) * this.speed;
            let newY = this.y + (dy / distance) * this.speed;

            if (this.canMoveTo(newX, newY, maze)) {
                this.x = newX;
                this.y = newY;
            } else {
                if (this.canMoveTo(newX, this.y, maze)) {
                    this.x = newX;
                } else if (this.canMoveTo(this.x, newY, maze)) {
                    this.y = newY;
                } else {
                    this.x = this.targetX;
                    this.y = this.targetY;
                }
            }
        } else {
            this.x = this.targetX;
            this.y = this.targetY;
        }

        // Check for power-up collection
        for (let i = powerUps.length - 1; i >= 0; i--) {
            if (this.collidesWith(powerUps[i])) {
                this.collectPowerUp();
                powerUps.splice(i, 1);
            }
        }

        // Update speed boost timer
        if (this.speedBoostTimer > 0) {
            this.speedBoostTimer--;
            if (this.speedBoostTimer === 0) {
                this.speed = this.baseSpeed;
            }
        }
    }

    draw(ctx) {
        ctx.fillStyle = this.dead ? '#ff0000' : (this.speedBoostTimer > 0 ? '#00FFFF' : '#ff0');
        ctx.fillRect(this.x, this.y, this.size, this.size);
    }

    isMoving() {
        return this.x !== this.targetX || this.y !== this.targetY;
    }

    canMoveTo(x, y, maze) {
        let currentCol = Math.floor(this.x / CELL_SIZE);
        let currentRow = Math.floor(this.y / CELL_SIZE);
        let targetCol = Math.floor(x / CELL_SIZE);
        let targetRow = Math.floor(y / CELL_SIZE);

        if (targetCol < 0 || targetCol >= maze.cols || targetRow < 0 || targetRow >= maze.rows) {
            return false;
        }

        if (currentCol === targetCol && currentRow === targetRow) {
            return true;
        }

        let currentCell = maze.grid[currentRow][currentCol];
        let targetCell = maze.grid[targetRow][targetCol];

        if (targetCol > currentCol && (currentCell.walls.right || targetCell.walls.left)) return false;
        if (targetCol < currentCol && (currentCell.walls.left || targetCell.walls.right)) return false;
        if (targetRow > currentRow && (currentCell.walls.bottom || targetCell.walls.top)) return false;
        if (targetRow < currentRow && (currentCell.walls.top || targetCell.walls.bottom)) return false;

        return true;
    }

    tryMove(dx, dy, maze) {
        let newCol = Math.floor((this.x + dx * CELL_SIZE + this.size / 2) / CELL_SIZE);
        let newRow = Math.floor((this.y + dy * CELL_SIZE + this.size / 2) / CELL_SIZE);

        if (newCol >= 0 && newCol < maze.cols && newRow >= 0 && newRow < maze.rows) {
            let currentCell = maze.grid[Math.floor(this.y / CELL_SIZE)][Math.floor(this.x / CELL_SIZE)];
            let targetCell = maze.grid[newRow][newCol];

            if ((dx === 1 && !currentCell.walls.right && !targetCell.walls.left) ||
                (dx === -1 && !currentCell.walls.left && !targetCell.walls.right) ||
                (dy === 1 && !currentCell.walls.bottom && !targetCell.walls.top) ||
                (dy === -1 && !currentCell.walls.top && !targetCell.walls.bottom)) {
                this.targetX = newCol * CELL_SIZE;
                this.targetY = newRow * CELL_SIZE;
            }
        }
    }

    setMoving(dx, dy, isMoving) {
        if (this.dead) return;

        this.moving = isMoving;
        if (isMoving) {
            this.currentDirection.dx = dx;
            this.currentDirection.dy = dy;
        } else {
            this.currentDirection.dx = 0;
            this.currentDirection.dy = 0;
        }
    }

    die() {
        this.dead = true;
        this.moving = false;
        this.currentDirection = { dx: 0, dy: 0 };
    }

    collidesWith(powerUp) {
        return (this.x < powerUp.x + powerUp.size &&
                this.x + this.size > powerUp.x &&
                this.y < powerUp.y + powerUp.size &&
                this.y + this.size > powerUp.y);
    }

    collectPowerUp() {
        this.speed = this.baseSpeed * 1.5;
        this.speedBoostTimer = 180; // 3 seconds at 60 FPS
    }
}

class Enemy {
    constructor(x, y, size) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.speed = 1.5;
        this.targetX = x;
        this.targetY = y;
        this.pathfindingCooldown = 0;
    }

    update(maze, player) {
        if (this.pathfindingCooldown <= 0) {
            this.findPathToPlayer(maze, player);
            this.pathfindingCooldown = 30;
        }
        this.pathfindingCooldown--;

        let dx = this.targetX - this.x;
        let dy = this.targetY - this.y;
        let distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > this.speed) {
            this.x += (dx / distance) * this.speed;
            this.y += (dy / distance) * this.speed;
        } else {
            this.x = this.targetX;
            this.y = this.targetY;
        }

        let playerDist = Math.sqrt(
            Math.pow(this.x - player.x, 2) +
            Math.pow(this.y - player.y, 2)
        );

        if (playerDist < this.size) {
            player.die();
        }
    }

    findPathToPlayer(maze, player) {
        let currentCol = Math.floor(this.x / CELL_SIZE);
        let currentRow = Math.floor(this.y / CELL_SIZE);
        let playerCol = Math.floor(player.x / CELL_SIZE);
        let playerRow = Math.floor(player.y / CELL_SIZE);

        let cell = maze.grid[currentRow][currentCol];

        let dx = 0;
        let dy = 0;

        if (playerCol > currentCol && !cell.walls.right) dx = 1;
        else if (playerCol < currentCol && !cell.walls.left) dx = -1;

        if (playerRow > currentRow && !cell.walls.bottom) dy = 1;
        else if (playerRow < currentRow && !cell.walls.top) dy = -1;

        if (dx === 0 && dy === 0) {
            if (!cell.walls.right) dx = 1;
            else if (!cell.walls.left) dx = -1;
            else if (!cell.walls.bottom) dy = 1;
            else if (!cell.walls.top) dy = -1;
        }

        this.targetX = (currentCol + dx) * CELL_SIZE;
        this.targetY = (currentRow + dy) * CELL_SIZE;
    }

    draw(ctx) {
        ctx.fillStyle = '#ff0000';
        ctx.beginPath();
        ctx.arc(
            this.x + this.size / 2,
            this.y + this.size / 2,
            this.size / 2,
            0,
            Math.PI * 2
        );
        ctx.fill();
    }
}