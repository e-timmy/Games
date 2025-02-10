class Game {
    constructor() {
        this.initializeElements();
        this.initializeGameState();
        this.initializeControls();
        this.initializeWeather();
        this.gameLoop(0);
    }

    initializeElements() {
        this.container = document.getElementById('gameContainer');
        this.player = document.getElementById('player');
        this.playerSpraycans = this.player.querySelector('.player-spraycans');
        this.sprayEffect = this.player.querySelector('.spray-effect');
        this.scoreElement = document.getElementById('score');
        this.gameOverElement = document.getElementById('gameOver');
    }

    initializeGameState() {
        // Game objects
        this.platforms = [];
        this.enemies = [];
        this.rockets = [];
        this.spraycans = [];
        this.collectedSpraycans = [];

        // Game settings
        this.platformWidth = 60;
        this.platformHeight = 8;
        this.initialPlatformSpacing = 60;
        this.screenHeight = 600;
        this.jumpHeight = this.screenHeight * 0.55;
        this.gravity = 0.25;
        this.breakablePlatformChance = 0.3;
        this.trampolineChance = 0.2;
        this.enemyChance = 0.1;
        this.rocketChance = 0.05;
        this.spraycanChance = 0.1;

        // Player state
        this.playerPos = { x: 200, y: 500 };
        this.playerVel = { x: 0, y: 0 };
        this.playerSize = { width: 20, height: 30 };
        this.cameraY = 0;
        this.maxCameraY = 0;
        this.rocketActive = false;
        this.rocketTimer = 0;
        this.sprayActive = false;
        this.sprayTimer = 0;
        this.isFlipping = false;
        this.flipTimer = 0;

        // Game state
        this.score = 0;
        this.gameActive = true;
        this.highestPlatformY = 550;
        this.lastPlatformWasBreakable = false;
        this.lastTime = 0;

        // Create initial platforms
        this.createPlatform(200, 550, false);
        for (let i = 0; i < 15; i++) {
            this.generatePlatform();
        }

        // Weather state
        this.currentWeather = 'clear';
        this.weatherTimer = 0;
        this.weatherDuration = 10000; // 10 seconds
        this.windStrength = 0;
        this.windParticles = [];
    }

    initializeWeather() {
        this.currentWeather = 'windy';
        this.windStrength = 2;  // Constant wind strength for testing
        console.log('Weather initialized: Windy');
    }

    initializeControls() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') this.playerVel.x = -6;
            if (e.key === 'ArrowRight') this.playerVel.x = 6;
            if (e.key === ' ') this.activateSpray();
        });

        document.addEventListener('keyup', (e) => {
            if (e.key === 'ArrowLeft' && this.playerVel.x < 0) this.playerVel.x = 0;
            if (e.key === 'ArrowRight' && this.playerVel.x > 0) this.playerVel.x = 0;
        });
    }

    startFlipAnimation() {
        if (!this.isFlipping) {
            this.isFlipping = true;
            this.flipTimer = 500; // 500ms flip duration
            this.player.classList.add('flipping');

            // Update CSS variables for the flip animation
            this.player.style.setProperty('--player-x', `${this.playerPos.x}px`);
            this.player.style.setProperty('--player-y', `${this.playerPos.y + this.cameraY}px`);

            // Remove the flipping class when animation ends
            setTimeout(() => {
                this.player.classList.remove('flipping');
                this.isFlipping = false;
            }, 500);
        }
    }

    activateSpray() {
        if (this.collectedSpraycans.length > 0 && !this.sprayActive) {
            const spraycan = this.collectedSpraycans.pop();
            this.sprayActive = true;
            this.sprayTimer = 1000;
            this.sprayEffect.style.backgroundColor = spraycan.color;
            this.sprayEffect.style.height = '100px';
            this.updatePlayerSpraycans();
            this.playerVel.y = -Math.sqrt(2 * this.gravity * this.jumpHeight);

            // Start the flip animation
            this.startFlipAnimation();
        }
    }

    createEnemy(x, y) {
        const enemy = document.createElement('div');
        enemy.className = 'enemy';

        const enemyInner = document.createElement('div');
        enemyInner.className = 'enemy-inner';
        enemy.appendChild(enemyInner);

        enemy.style.transform = `translate(${x}px, ${y}px)`;
        this.container.appendChild(enemy);

        this.enemies.push({
            element: enemy,
            x,
            y,
            direction: 1,
            speed: 2,
            active: true
        });
    }

    createRocket(x, y) {
        const rocket = document.createElement('div');
        rocket.className = 'rocket';
        rocket.style.transform = `translate(${x}px, ${y}px)`;
        this.container.appendChild(rocket);

        this.rockets.push({
            element: rocket,
            x,
            y
        });
    }

    createSpraycan(x, y) {
        const spraycan = document.createElement('div');
        spraycan.className = 'spraycan';
        const color = this.getRandomColor();
        spraycan.style.backgroundColor = color;
        spraycan.style.transform = `translate(${x}px, ${y}px)`;
        this.container.appendChild(spraycan);

        this.spraycans.push({
            element: spraycan,
            x,
            y,
            color
        });
    }

    getRandomColor() {
        const hue = Math.floor(Math.random() * 360);
        return `hsl(${hue}, 100%, 50%)`;
    }

    createPlatform(x, y, breakable = false) {
        const platform = document.createElement('div');
        platform.className = 'platform' + (breakable ? ' breakable' : '');
        platform.style.width = this.platformWidth + 'px';
        platform.style.transform = `translate(${x}px, ${y}px)`;

        const hasTrampoline = !breakable && Math.random() < this.trampolineChance;
        if (hasTrampoline) {
            const trampoline = document.createElement('div');
            trampoline.className = 'trampoline';
            platform.appendChild(trampoline);
        }

        this.container.appendChild(platform);
        this.platforms.push({
            element: platform,
            x,
            y,
            breakable,
            hasTrampoline,
            active: true
        });

        this.highestPlatformY = Math.min(this.highestPlatformY, y);

        // Create power-ups and enemies
        if (Math.random() < this.enemyChance) {
            this.createEnemy(x + this.platformWidth / 2, y - 40);
        }
        if (Math.random() < this.rocketChance) {
            this.createRocket(x + this.platformWidth / 2, y - 40);
        }
        if (Math.random() < this.spraycanChance) {
            this.createSpraycan(x + this.platformWidth / 2, y - 40);
        }
    }

    generatePlatform() {
        const spacing = this.getPlatformSpacing();
        const x = Math.random() * (400 - this.platformWidth);
        const y = this.highestPlatformY - spacing;

        let makeBreakable = Math.random() < this.breakablePlatformChance;
        if (this.lastPlatformWasBreakable) {
            makeBreakable = false;
        }

        this.createPlatform(x, y, makeBreakable);
        this.lastPlatformWasBreakable = makeBreakable;
    }

    breakPlatform(platform) {
        if (!platform.active) return;

        platform.active = false;
        platform.element.classList.add('breaking');
        platform.element.style.setProperty('--camera-y', `${this.cameraY}px`);

        setTimeout(() => {
            if (platform.element.parentNode) {
                platform.element.remove();
            }
            const index = this.platforms.indexOf(platform);
            if (index > -1) {
                this.platforms.splice(index, 1);
            }
        }, 300);
    }

    updateWeather(deltaTime) {
        // Always windy for testing
        this.currentWeather = 'windy';
        this.windStrength = 2;  // Constant wind strength for testing
        this.updateWindParticles();
    }

    updateWindParticles() {
        // Create new particles
        if (Math.random() < 0.3) {
            const particle = document.createElement('div');
            particle.className = 'wind-particle';
            particle.style.top = `${Math.random() * 600}px`;
            particle.style.left = this.windStrength > 0 ? '-20px' : '420px';
            this.container.appendChild(particle);
            this.windParticles.push({
                element: particle,
                x: this.windStrength > 0 ? -20 : 420,
                y: Math.random() * 600
            });
        }

        // Update particle positions
        for (let i = this.windParticles.length - 1; i >= 0; i--) {
            const particle = this.windParticles[i];
            particle.x += this.windStrength * 2;
            particle.element.style.transform = `translate(${particle.x}px, ${particle.y + this.cameraY}px)`;

            // Remove particles that are out of bounds
            if (particle.x < -20 || particle.x > 420) {
                particle.element.remove();
                this.windParticles.splice(i, 1);
            }
        }
    }

    updateEnemies() {
        for (const enemy of this.enemies) {
            if (!enemy.active) continue;

            enemy.x += enemy.speed * enemy.direction + this.windStrength * 0.5;
            if (enemy.x <= 0 || enemy.x >= 380) {
                enemy.direction *= -1;
            }

            enemy.element.style.transform = `translate(${enemy.x}px, ${enemy.y + this.cameraY}px)`;

            if (!this.rocketActive && this.checkCollision(
                enemy.x, enemy.y, 20, 20,
                this.playerPos.x, this.playerPos.y, this.playerSize.width, this.playerSize.height
            )) {
                this.gameOver();
                return;
            }
        }
    }

    updateRockets() {
        for (let i = this.rockets.length - 1; i >= 0; i--) {
            const rocket = this.rockets[i];
            rocket.element.style.transform = `translate(${rocket.x}px, ${rocket.y + this.cameraY}px)`;

            if (this.checkCollision(
                rocket.x, rocket.y, 20, 40,
                this.playerPos.x, this.playerPos.y, this.playerSize.width, this.playerSize.height
            )) {
                this.rocketActive = true;
                this.rocketTimer = 2000;
                this.player.classList.add('with-rocket');
                rocket.element.remove();
                this.rockets.splice(i, 1);
            }
        }
    }

    updateSpraycans() {
        for (let i = this.spraycans.length - 1; i >= 0; i--) {
            const spraycan = this.spraycans[i];
            spraycan.element.style.transform = `translate(${spraycan.x}px, ${spraycan.y + this.cameraY}px)`;

            if (this.checkCollision(
                spraycan.x, spraycan.y, 10, 15,
                this.playerPos.x, this.playerPos.y, this.playerSize.width, this.playerSize.height
            )) {
                if (this.collectedSpraycans.length < 3) {
                    this.collectedSpraycans.push(spraycan);
                    this.updatePlayerSpraycans();
                    spraycan.element.remove();
                    this.spraycans.splice(i, 1);
                }
            }
        }
    }

    updatePlayerSpraycans() {
        this.playerSpraycans.innerHTML = '';
        this.collectedSpraycans.forEach(spraycan => {
            const spraycanElement = document.createElement('div');
            spraycanElement.className = 'player-spraycan';
            spraycanElement.style.backgroundColor = spraycan.color;
            this.playerSpraycans.appendChild(spraycanElement);
        });
    }

    checkCollision(x1, y1, w1, h1, x2, y2, w2, h2) {
        return (
            x1 < x2 + w2 &&
            x1 + w1 > x2 &&
            y1 < y2 + h2 &&
            y1 + h1 > y2
        );
    }

    cleanupElements() {
        const cleanup = (array, isActive = true) => {
            return array.filter(item => {
                if ((!isActive || item.active) && item.y + this.cameraY > 700) {
                    if (item.element.parentNode) {
                        item.element.remove();
                    }
                    return false;
                }
                return true;
            });
        };

        this.platforms = cleanup(this.platforms);
        this.enemies = cleanup(this.enemies);
        this.rockets = cleanup(this.rockets);
        this.spraycans = cleanup(this.spraycans);
    }

    update(deltaTime) {
        if (!this.gameActive) return;

        this.updateWeather(deltaTime);

        if (this.rocketActive) {
            this.rocketTimer -= deltaTime;
            if (this.rocketTimer <= 0) {
                this.rocketActive = false;
                this.player.classList.remove('with-rocket');
            } else {
                this.playerVel.y = -10;
            }
        } else if (this.sprayActive) {
            this.sprayTimer -= deltaTime;
            if (this.sprayTimer <= 0) {
                this.sprayActive = false;
                this.sprayEffect.style.height = '0';
            }
        } else {
            this.playerVel.y += this.gravity;
        }

        // Update flip timer
        if (this.isFlipping) {
            this.flipTimer -= deltaTime;
            if (this.flipTimer <= 0) {
                this.isFlipping = false;
            }
        }

        this.playerPos.x += this.playerVel.x + this.windStrength * 0.1;
        this.playerPos.y += this.playerVel.y;

        if (this.playerPos.x < 0) this.playerPos.x = 0;
        if (this.playerPos.x > 400 - this.playerSize.width) {
            this.playerPos.x = 400 - this.playerSize.width;
        }

        const idealCameraY = -this.playerPos.y + 400;
        if (idealCameraY < this.maxCameraY) {
            this.cameraY = this.maxCameraY;
        } else {
            this.cameraY = idealCameraY;
            this.maxCameraY = idealCameraY;
        }

        if (this.playerPos.y + this.cameraY > 600) {
            this.gameOver();
            return;
        }

        this.updatePlatforms();
        this.updateEnemies();
        this.updateRockets();
        this.updateSpraycans();

        this.ensurePlatforms();
        this.cleanupElements();

        // Only update transform if not flipping
        if (!this.isFlipping) {
            this.player.style.transform = `translate(${this.playerPos.x}px, ${this.playerPos.y + this.cameraY}px)`;
        }
    }

    updatePlatforms() {
        for (const platform of this.platforms) {
            if (!platform.active) continue;

            platform.element.style.transform = `translate(${platform.x}px, ${platform.y + this.cameraY}px)`;

            if (this.playerVel.y > 0 && this.checkCollision(
                platform.x, platform.y, this.platformWidth, this.platformHeight,
                this.playerPos.x,
                this.playerPos.y, this.playerSize.width, this.playerSize.height
            )) {
                if (platform.breakable) {
                    this.breakPlatform(platform);
                } else {
                    this.playerPos.y = platform.y - this.playerSize.height;
                    const jumpHeight = platform.hasTrampoline ? this.jumpHeight * 2 : this.jumpHeight;
                    this.playerVel.y = -Math.sqrt(2 * this.gravity * jumpHeight);
                    this.score++;
                    this.scoreElement.textContent = this.score;
                }
            }
        }
    }

    ensurePlatforms() {
        const visibleTop = -this.cameraY;
        while (this.highestPlatformY > visibleTop - 400) {
            this.generatePlatform();
        }
    }

    getPlatformSpacing() {
        const maxSpacing = 200;
        const minSpacing = this.initialPlatformSpacing;
        const difficultyFactor = Math.min(this.score / 100, 1);
        return minSpacing + (maxSpacing - minSpacing) * difficultyFactor;
    }

    gameLoop(timestamp) {
        const deltaTime = timestamp - this.lastTime;
        this.lastTime = timestamp;

        this.update(deltaTime);
        requestAnimationFrame((t) => this.gameLoop(t));
    }

    gameOver() {
        this.gameActive = false;
        this.gameOverElement.style.display = 'block';
        document.getElementById('finalScore').textContent = this.score;
    }
}

// Start the game
new Game();