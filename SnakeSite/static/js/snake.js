class SnakeGame {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.scoreElement = document.querySelector('#score span');
        this.navItems = document.querySelectorAll('.nav-item');
        this.contentSections = document.querySelectorAll('.content-section');

        this.setupCanvas();
        this.bindEvents();
        this.initGame();

        // Add click handlers for nav items
        this.navItems.forEach(item => {
            item.addEventListener('click', () => {
                this.activateSection(item.dataset.section);
            });
        });

        this.particles = new ParticleSystem(this.canvas, this);
        this.currentSection = 'home';
    }

    setupCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.gridSize = 20;
        this.tileCountX = Math.floor(this.canvas.width / this.gridSize);
        this.tileCountY = Math.floor(this.canvas.height / this.gridSize);
    }

    bindEvents() {
        window.addEventListener('keydown', this.handleKeyPress.bind(this));
        window.addEventListener('resize', () => {
            this.setupCanvas();
        });
    }

    initGame() {
        this.snake = [{x: 10, y: 10}];
        this.dx = 1;
        this.dy = 0;
        this.score = 0;
        this.speed = 150;
        this.lastRenderTime = 0;
        this.food = this.generateFood();
        this.animate(0);
    }

    handleKeyPress(e) {
        if(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
            e.preventDefault();
        }

        switch(e.key) {
            case 'ArrowUp':
                if (this.dy === 0) { this.dx = 0; this.dy = -1; }
                break;
            case 'ArrowDown':
                if (this.dy === 0) { this.dx = 0; this.dy = 1; }
                break;
            case 'ArrowLeft':
                if (this.dx === 0) { this.dx = -1; this.dy = 0; }
                break;
            case 'ArrowRight':
                if (this.dx === 0) { this.dx = 1; this.dy = 0; }
                break;
        }
    }

    generateFood() {
        return {
            x: Math.floor(Math.random() * this.tileCountX),
            y: Math.floor(Math.random() * this.tileCountY)
        };
    }

    animate(currentTime) {
        window.requestAnimationFrame(this.animate.bind(this));

        const secondsSinceLastRender = (currentTime - this.lastRenderTime) / 1000;
        if (secondsSinceLastRender < 1 / (this.speed / 10)) return;

        this.lastRenderTime = currentTime;
        this.update();
        this.draw();
    }

    update() {
        const head = {
            x: (this.snake[0].x + this.dx + this.tileCountX) % this.tileCountX,
            y: (this.snake[0].y + this.dy + this.tileCountY) % this.tileCountY
        };

        this.snake.unshift(head);

        if (head.x === this.food.x && head.y === this.food.y) {
            this.score += 10;
            this.scoreElement.textContent = this.score;
            this.food = this.generateFood();
            this.speed = Math.min(this.speed + 5, 300);
        } else {
            this.snake.pop();
        }

        this.checkNavCollisions(head);
    }

    checkNavCollisions(head) {
        this.navItems.forEach((item, index) => {
            const rect = item.getBoundingClientRect();
            const snakeX = head.x * this.gridSize;
            const snakeY = head.y * this.gridSize;

            if (snakeX >= rect.left && snakeX <= rect.right &&
                snakeY >= rect.top && snakeY <= rect.bottom) {
                this.activateSection(item.dataset.section);
            }
        });
    }

    activateSection(sectionId) {
        this.currentSection = sectionId;

        // Remove active class from all nav items and sections
        this.navItems.forEach(item => item.classList.remove('active'));
        this.contentSections.forEach(section => section.classList.remove('active'));

        // Add active class to current nav item and section
        document.querySelector(`.nav-item[data-section="${sectionId}"]`).classList.add('active');
        document.getElementById(sectionId).classList.add('active');
    }

    draw() {
        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Update and draw particles
        this.particles.update(this.currentSection);
        this.particles.draw();

        // Draw snake
        this.snake.forEach((segment, index) => {
            const size = this.gridSize - 1;
            const x = segment.x * this.gridSize;
            const y = segment.y * this.gridSize;

            this.ctx.fillStyle = index === 0 ? '#8f94ff' : '#646cff';
            this.ctx.fillRect(x, y, size, size);
        });

        // Draw food
        this.ctx.fillStyle = '#ff4646';
        const foodSize = this.gridSize - 1;
        this.ctx.fillRect(
            this.food.x * this.gridSize,
            this.food.y * this.gridSize,
            foodSize,
            foodSize
        );
    }
}

window.addEventListener('load', () => {
    new SnakeGame();
});