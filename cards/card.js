class Card {
    constructor(suit, value) {
        this.suit = suit;
        this.value = value;
        this.faceDown = true;
        this.element = this.createCardElement();
        this.isAnimating = false;
    }

    get displayValue() {
        const valueMap = {
            1: 'A',
            11: 'J',
            12: 'Q',
            13: 'K'
        };
        return valueMap[this.value] || this.value;
    }

    get suitSymbol() {
        const suitSymbols = {
            'hearts': '♥',
            'diamonds': '♦',
            'clubs': '♣',
            'spades': '♠'
        };
        return suitSymbols[this.suit];
    }

    get isRed() {
        return this.suit === 'hearts' || this.suit === 'diamonds';
    }

    createCardElement() {
        const card = document.createElement('div');
        card.className = 'card face-down';
        card.style.position = 'absolute';
        card.style.transform = 'translate(0, 0) rotateY(180deg)';

        const front = document.createElement('div');
        front.className = 'card-front';
        this.renderCardFront(front);

        const back = document.createElement('div');
        back.className = 'card-back';

        card.appendChild(front);
        card.appendChild(back);

        return card;
    }

    renderCardFront(element) {
        element.style.color = this.isRed ? '#ff0000' : '#000000';

        const topLeft = document.createElement('div');
        topLeft.className = 'card-corner top-left';
        topLeft.innerHTML = `${this.displayValue}<br>${this.suitSymbol}`;

        const bottomRight = document.createElement('div');
        bottomRight.className = 'card-corner bottom-right';
        bottomRight.innerHTML = `${this.displayValue}<br>${this.suitSymbol}`;

        const center = document.createElement('div');
        center.className = 'card-center';
        center.textContent = this.suitSymbol;

        element.appendChild(topLeft);
        element.appendChild(center);
        element.appendChild(bottomRight);
    }

    flip() {
        if (this.isAnimating) return;

        this.faceDown = !this.faceDown;
        this.isAnimating = true;

        return new Promise(resolve => {
            this.element.style.transition = 'transform 0.3s ease-out';
            this.element.style.transform = `translate(${this.element.style.getPropertyValue('--dealX') || '0px'}, ${this.element.style.getPropertyValue('--dealY') || '0px'}) rotateY(${this.faceDown ? '180deg' : '0deg'})`;

            setTimeout(() => {
                this.isAnimating = false;
                resolve();
            }, 300);
        });
    }

    async animateDeal(x, y, delay = 0) {
        if (this.isAnimating) return;

        this.isAnimating = true;
        console.log(`Animating card to position: (${x}, ${y})`);

        // Set initial position
        this.element.style.setProperty('--dealX', `${x}px`);
        this.element.style.setProperty('--dealY', `${y}px`);

        return new Promise(resolve => {
            setTimeout(async () => {
                // Slide animation
                this.element.style.transition = 'transform 0.5s ease-out';
                this.element.style.transform = `translate(${x}px, ${y}px) rotateY(180deg)`;

                setTimeout(async () => {
                    this.isAnimating = false;
                    resolve();
                }, 500);
            }, delay);
        });
    }

    getGameValue() {
        if (this.value === 1) return 11;  // Ace
        if (this.value > 10) return 10;   // Face cards
        return this.value;
    }
}