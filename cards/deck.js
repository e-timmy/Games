class Deck {
    constructor() {
        this.cards = [];
        this.init();
    }

    init() {
        const suits = ['hearts', 'diamonds', 'clubs', 'spades'];
        const values = Array.from({length: 13}, (_, i) => i + 1);

        for (let suit of suits) {
            for (let value of values) {
                this.cards.push(new Card(suit, value));
            }
        }

        this.shuffle();
    }

    shuffle() {
        for (let i = this.cards.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.cards[i], this.cards[j]] = [this.cards[j], this.cards[i]];
        }
    }

    deal() {
        if (this.cards.length === 0) {
            this.init();
        }
        return this.cards.pop();
    }

    reset() {
        this.cards.forEach(card => {
            if (card.element.parentNode) {
                card.element.parentNode.removeChild(card.element);
            }
        });
        this.init();
    }
}
