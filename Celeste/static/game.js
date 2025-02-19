class Game {
    constructor() {
        this.canvas = document.getElementById('game-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.sessionId = Math.random().toString(36).substr(2, 9);

        this.keys = {
            left: 0,
            right: 0,
            up: 0,
            down: 0,
            z: 0,
            x: 0,
            c: 0
        };

        this.sparkles = Array(50).fill().map(() => ({
            x: Math.random() * this.canvas.width,
            y: Math.random() * this.canvas.height,
            size: Math.random() * 2 + 1
        }));

        this.lastUpdateTime = performance.now();
        this.fixedTimeStep = 1000 / 60; // 60 FPS
        this.accumulatedTime = 0;

        this.setupEventListeners();
        this.gameLoop();
    }

    setupEventListeners() {
        document.addEventListener('keydown', (e) => this.handleKeyEvent(e, 1));
        document.addEventListener('keyup', (e) => this.handleKeyEvent(e, 0));
    }

    handleKeyEvent(e, value) {
        e.preventDefault(); // Prevent default browser behavior
        switch(e.key.toLowerCase()) {
            case 'arrowleft':
                this.keys.left = value;
                break;
            case 'arrowright':
                this.keys.right = value;
                break;
            case 'arrowup':
                this.keys.up = value;
                break;
            case 'arrowdown':
                this.keys.down = value;
                break;
            case 'z':
                this.keys.z = value;
                break;
            case 'x':
                this.keys.x = value;
                break;
            case 'c':
                this.keys.c = value;
                break;
        }
        console.log('Key state updated:', this.keys);  // Debug log
    }

    async updateGameState() {
        try {
            const response = await fetch('/api/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    input: this.keys
                })
            });
            const data = await response.json();
            console.log('Received game state:', data);  // Debug log
            return data;
        } catch (error) {
            console.error('Error updating game state:', error);
            return null;
        }
    }

    async gameLoop() {
        const gameState = await this.updateGameState();

        if (gameState) {
            this.render(gameState);

            if (gameState.win) {
                // Reset game after a short delay when won
                setTimeout(async () => {
                    await fetch('/api/reset', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            session_id: this.sessionId
                        })
                    });
                }, 1000);
            }
        }

        requestAnimationFrame(() => this.gameLoop());
    }

    drawBackground() {
        // Draw gradient background
        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
        gradient.addColorStop(0, '#142850');
        gradient.addColorStop(1, '#27496d');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Update and draw sparkles
        this.sparkles.forEach(sparkle => {
            sparkle.x += (Math.random() - 0.5) * 2;
            sparkle.y += (Math.random() - 0.5) * 2;

            if (sparkle.x < 0) sparkle.x = this.canvas.width;
            if (sparkle.x > this.canvas.width) sparkle.x = 0;
            if (sparkle.y < 0) sparkle.y = this.canvas.height;
            if (sparkle.y > this.canvas.height) sparkle.y = 0;

            const alpha = Math.random() * 0.5 + 0.5;
            this.ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.arc(sparkle.x, sparkle.y, sparkle.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    drawPlatforms(platforms) {
        platforms.forEach(platform => {
            // Platform base
            this.ctx.fillStyle = '#503c78';
            this.ctx.fillRect(platform.x, platform.y, platform.width, platform.height);

            // Platform glow
            this.ctx.strokeStyle = '#7864b4';
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(platform.x, platform.y, platform.width, platform.height);
        });
    }

    drawPlayer(player) {
        // Player trail
        const trailLength = 10;
        const trailOpacityStep = 1 / trailLength;

        for (let i = 0; i < trailLength; i++) {
            const trailX = player.x - player.vel_x * i * 0.5;
            const trailY = player.y - player.vel_y * i * 0.5;
            const opacity = 1 - (i * trailOpacityStep);

            this.ctx.fillStyle = `rgba(200, 200, 255, ${opacity * 0.3})`;
            this.ctx.beginPath();
            this.ctx.arc(
                trailX + player.width / 2,
                trailY + player.height / 2,
                5,
                0,
                Math.PI * 2
            );
            this.ctx.fill();
        }

        // Player body
        let playerColor = player.is_boosting ? '#ffa500' :
                         player.is_climbing ? '#32cd32' :
                         '#c8c8ff';

        this.ctx.fillStyle = playerColor;
        this.ctx.fillRect(player.x, player.y, player.width, player.height);

        // Player glow
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(player.x, player.y, player.width, player.height);
    }

    drawWinZone() {
        // Win zone position (top right)
        const winZone = {
            x: this.canvas.width - 80,
            y: 0,
            width: 80,
            height: 100
        };

        // Pulsing effect
        const pulse = Math.abs(Math.sin(Date.now() / 500));
        const alpha = 0.3 + pulse * 0.2;

        this.ctx.fillStyle = `rgba(147, 0, 211, ${alpha})`;
        this.ctx.fillRect(winZone.x, winZone.y, winZone.width, winZone.height);

        // Win zone glow
        this.ctx.strokeStyle = `rgba(200, 100, 255, ${0.5 + pulse * 0.5})`;
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(winZone.x, winZone.y, winZone.width, winZone.height);
    }

    render(gameState) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        this.drawBackground();
        this.drawWinZone();
        this.drawPlatforms(gameState.platforms);
        this.drawPlayer(gameState.player);
    }

    async gameLoop() {
        const gameState = await this.updateGameState();

        if (gameState) {
            this.render(gameState);

            if (gameState.win) {
                // Reset game after a short delay when won
                setTimeout(async () => {
                    await fetch('/api/reset', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            session_id: this.sessionId
                        })
                    });
                }, 1000);
            }
        }

        requestAnimationFrame(() => this.gameLoop());
    }
}

// Start the game when the page loads
window.addEventListener('load', () => {
    new Game();
});