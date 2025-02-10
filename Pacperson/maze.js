class Cell {
    constructor(row, col) {
        this.row = row;
        this.col = col;
        this.walls = {
            top: true,
            right: true,
            bottom: true,
            left: true
        };
        this.visited = false;
        this.isStart = false;
        this.isEnd = false;
    }
}

class Maze {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.grid = [];
        this.stack = [];
        this.startCell = null;
        this.endCell = null;

        this.generateGrid();
        this.generateMaze();
        this.createOpenSpaces();
        this.setStartAndEndCells();
    }

    generateGrid() {
        for (let r = 0; r < this.rows; r++) {
            let row = [];
            for (let c = 0; c < this.cols; c++) {
                row.push(new Cell(r, c));
            }
            this.grid.push(row);
        }
    }

    generateMaze() {
        let current = this.grid[0][0];
        current.visited = true;
        this.stack.push(current);

        while (this.stack.length > 0) {
            current = this.stack.pop();
            let neighbor = this.getRandomUnvisitedNeighbor(current);

            if (neighbor) {
                this.stack.push(current);
                this.removeWalls(current, neighbor);
                neighbor.visited = true;
                this.stack.push(neighbor);
            }
        }
    }

    getRandomUnvisitedNeighbor(cell) {
        let neighbors = [];

        let top = cell.row > 0 ? this.grid[cell.row - 1][cell.col] : null;
        let right = cell.col < this.cols - 1 ? this.grid[cell.row][cell.col + 1] : null;
        let bottom = cell.row < this.rows - 1 ? this.grid[cell.row + 1][cell.col] : null;
        let left = cell.col > 0 ? this.grid[cell.row][cell.col - 1] : null;

        if (top && !top.visited) neighbors.push(top);
        if (right && !right.visited) neighbors.push(right);
        if (bottom && !bottom.visited) neighbors.push(bottom);
        if (left && !left.visited) neighbors.push(left);

        if (neighbors.length > 0) {
            return neighbors[Math.floor(Math.random() * neighbors.length)];
        }
        return null;
    }

    removeWalls(cell1, cell2) {
        let x = cell1.col - cell2.col;
        if (x === 1) {
            cell1.walls.left = false;
            cell2.walls.right = false;
        } else if (x === -1) {
            cell1.walls.right = false;
            cell2.walls.left = false;
        }

        let y = cell1.row - cell2.row;
        if (y === 1) {
            cell1.walls.top = false;
            cell2.walls.bottom = false;
        } else if (y === -1) {
            cell1.walls.bottom = false;
            cell2.walls.top = false;
        }
    }

    createOpenSpaces() {
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                if (Math.random() < 0.4) {  // 40% chance to create an open space
                    let cell = this.grid[r][c];
                    cell.walls = {top: false, right: false, bottom: false, left: false};
                }
            }
        }
    }

    setStartAndEndCells() {
        this.startCell = this.grid[0][0];
        this.startCell.isStart = true;

        let endRow = this.rows - 1;
        let endCol = this.cols - 1;
        this.endCell = this.grid[endRow][endCol];
        this.endCell.isEnd = true;
    }

    draw(ctx, cellSize) {
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                let cell = this.grid[r][c];
                let x = c * cellSize;
                let y = r * cellSize;

                if (cell.isStart) {
                    ctx.fillStyle = '#00ff00';
                    ctx.fillRect(x, y, cellSize, cellSize);
                } else if (cell.isEnd) {
                    ctx.fillStyle = '#ff0000';
                    ctx.fillRect(x, y, cellSize, cellSize);
                }

                ctx.strokeStyle = '#0f0';
                ctx.lineWidth = 2;

                if (cell.walls.top) {
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + cellSize, y);
                    ctx.stroke();
                }
                if (cell.walls.right) {
                    ctx.beginPath();
                    ctx.moveTo(x + cellSize, y);
                    ctx.lineTo(x + cellSize, y + cellSize);
                    ctx.stroke();
                }
                if (cell.walls.bottom) {
                    ctx.beginPath();
                    ctx.moveTo(x + cellSize, y + cellSize);
                    ctx.lineTo(x, y + cellSize);
                    ctx.stroke();
                }
                if (cell.walls.left) {
                    ctx.beginPath();
                    ctx.moveTo(x, y + cellSize);
                    ctx.lineTo(x, y);
                    ctx.stroke();
                }
            }
        }
    }
}