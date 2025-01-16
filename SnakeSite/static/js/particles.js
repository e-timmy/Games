class ParticleSystem {
    constructor(canvas, snake) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.snake = snake;
        this.particles = [];
        this.colors = {
            home: '#646cff',
            about: '#ff4646',
            projects: '#46ff64',
            contact: '#ffff46'
        };
        this.currentColor = this.colors.home;
        this.mouseX = this.canvas.width / 2;
        this.mouseY = this.canvas.height / 2;

        window.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
        });

        this.init();
    }

    init() {
        const particleCount = 100;
        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                size: Math.random() * 3 + 1,
                baseSize: Math.random() * 3 + 1,
                speedX: Math.random() * 2 - 1,
                speedY: Math.random() * 2 - 1,
                brightness: 0.2
            });
        }
    }

    update(activeSection) {
        const targetColor = this.colors[activeSection] || this.colors.home;
        this.currentColor = this.lerpColor(this.currentColor, targetColor, 0.05);

        const snakeHead = this.snake.snake[0];
        const snakeX = snakeHead.x * this.snake.gridSize;
        const snakeY = snakeHead.y * this.snake.gridSize;

        // Calculate canvas diagonal for normalized distances
        const canvasDiagonal = Math.sqrt(
            this.canvas.width * this.canvas.width +
            this.canvas.height * this.canvas.height
        );

        this.particles.forEach(particle => {
            // Movement code
            particle.x += particle.speedX;
            particle.y += particle.speedY;

            // Wrap around screen
            if (particle.x > this.canvas.width) particle.x = 0;
            if (particle.x < 0) particle.x = this.canvas.width;
            if (particle.y > this.canvas.height) particle.y = 0;
            if (particle.y < 0) particle.y = this.canvas.height;

            // React to snake
            const dxSnake = snakeX - particle.x;
            const dySnake = snakeY - particle.y;
            const distanceSnake = Math.sqrt(dxSnake * dxSnake + dySnake * dySnake);

            if (distanceSnake < 100) {
                const angle = Math.atan2(dySnake, dxSnake);
                particle.speedX -= Math.cos(angle) * 0.2;
                particle.speedY -= Math.sin(angle) * 0.2;
            }

            // Calculate distance to mouse
            const dxMouse = this.mouseX - particle.x;
            const dyMouse = this.mouseY - particle.y;
            const distanceToMouse = Math.sqrt(dxMouse * dxMouse + dyMouse * dyMouse);

            // Normalize distance (0 to 1)
            const normalizedDistance = distanceToMouse / canvasDiagonal;

            // Calculate brightness using a smooth falloff
            const minBrightness = 0.2;
            const maxBrightness = 1.0;
            const brightnessRange = maxBrightness - minBrightness;

            // Inverse square falloff for more gradual effect
            particle.brightness = maxBrightness - (brightnessRange * normalizedDistance * normalizedDistance);

            // Ensure minimum brightness
            particle.brightness = Math.max(minBrightness, particle.brightness);

            // Adjust size based on brightness
            particle.size = particle.baseSize * (1 + (particle.brightness - minBrightness) * 0.5);

            // Add some randomness to movement
            particle.speedX += (Math.random() - 0.5) * 0.1;
            particle.speedY += (Math.random() - 0.5) * 0.1;

            // Limit speed
            const speed = Math.sqrt(particle.speedX * particle.speedX + particle.speedY * particle.speedY);
            if (speed > 2) {
                particle.speedX = (particle.speedX / speed) * 2;
                particle.speedY = (particle.speedY / speed) * 2;
            }
        });
    }

    draw() {
        this.particles.forEach(particle => {
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);

            const rgb = this.hexToRgb(this.currentColor);
            const brightnessMultiplier = 1 + particle.brightness;

            this.ctx.fillStyle = `rgba(
                ${Math.min(255, rgb.r * brightnessMultiplier)}, 
                ${Math.min(255, rgb.g * brightnessMultiplier)}, 
                ${Math.min(255, rgb.b * brightnessMultiplier)}, 
                ${particle.brightness}
            )`;
            this.ctx.fill();
        });
    }

    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    lerpColor(color1, color2, amount) {
        const rgb1 = this.hexToRgb(color1);
        const rgb2 = this.hexToRgb(color2);

        const r = Math.round(rgb1.r + (rgb2.r - rgb1.r) * amount);
        const g = Math.round(rgb1.g + (rgb2.g - rgb1.g) * amount);
        const b = Math.round(rgb1.b + (rgb2.b - rgb1.b) * amount);

        return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
    }
}