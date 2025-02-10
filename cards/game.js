class BlackjackGame {
    constructor() {
        this.deck = new Deck();
        this.playerCards = [];
        this.dealerCards = [];
        this.gameState = 'betting'; // betting, playing, dealer-turn, ended
        this.playerChips = { 5: 20, 25: 10, 100: 5, 500: 1 }; // Start with $1000 in chips
        this.currentBet = 0;
        this.betChips = [];
        this.sessionWinnings = 0; // Track session winnings

        // Initialize betting system
        this.setupChipDragging();
        this.updateBalanceDisplay();

        this.dealerArea = document.querySelector('.dealer-cards');
        this.playerArea = document.querySelector('.player-cards');
        this.dealerScore = document.querySelector('.dealer-score');
        this.playerScore = document.querySelector('.player-score');
        this.message = document.querySelector('.message');

        this.setupButtons();
        this.displayStartMessage();
    }

    displayStartMessage() {
        this.message.textContent = 'Place your bets to begin';
        this.message.classList.add('visible');
        setTimeout(() => {
            this.message.classList.remove('visible');
        }, 2000);
    }

    clearCards() {
        // Remove all card elements from the DOM
        while (this.dealerArea.firstChild) {
            this.dealerArea.removeChild(this.dealerArea.firstChild);
        }
        while (this.playerArea.firstChild) {
            this.playerArea.removeChild(this.playerArea.firstChild);
        }

        // Reset card arrays
        this.playerCards = [];
        this.dealerCards = [];

        // Reset scores
        this.updateScores();
    }

    endGame(message, isWin = false, isPush = false) {
        this.gameState = 'ended';
        let winAmount = 0;
        let resultMessage = message;

        // Calculate winnings and update message
        if (isWin) {
            if (message.includes("Blackjack")) {
                winAmount = Math.floor(this.currentBet * 2.5);
                resultMessage += ` (+$${winAmount})`;
            } else {
                winAmount = this.currentBet * 2;
                resultMessage += ` (+$${winAmount})`;
            }
            this.sessionWinnings += (winAmount - this.currentBet);
        } else if (isPush) {
            winAmount = this.currentBet;
            resultMessage += ` ($${winAmount} returned)`;
        } else {
            this.sessionWinnings -= this.currentBet;
            resultMessage += ` (-$${this.currentBet})`;
        }

        // Update session stats in message
        resultMessage += `\nSession: ${this.sessionWinnings >= 0 ? '+' : ''}$${this.sessionWinnings}`;

        this.message.textContent = resultMessage;
        this.message.classList.add('visible');

        // Handle payouts
        if (winAmount > 0) {
            this.payoutWinnings(winAmount);
        }

        // Disable game buttons
        document.getElementById('hit-button').disabled = true;
        document.getElementById('stay-button').disabled = true;

        // Reset for next hand after delay
        setTimeout(() => {
            this.message.classList.remove('visible');
            setTimeout(() => {
                this.clearCards();
                this.currentBet = 0;
                this.clearBettingArea();
                this.gameState = 'betting';
                document.getElementById('deal-button').disabled = false;
                this.displayStartMessage();
            }, 300);
        }, 3000);
    }

    determineWinner() {
        const playerScore = this.calculateScore(this.playerCards);
        const dealerScore = this.calculateScore(this.dealerCards);

        if (dealerScore > 21) {
            this.endGame("Dealer busts! Player wins!", true);
        } else if (dealerScore > playerScore) {
            this.endGame("Dealer wins!");
        } else if (playerScore > dealerScore) {
            this.endGame("Player wins!", true);
        } else {
            this.endGame("Push! It's a tie!", false, true);
        }
    }

    checkForBlackjack() {
        const playerScore = this.calculateScore(this.playerCards);

        if (playerScore === 21) {
            this.dealerCards[1].flip();
            const dealerScore = this.calculateScore(this.dealerCards);

            if (dealerScore === 21) {
                this.endGame("Push! Both have Blackjack!", false, true);
            } else {
                this.endGame("Blackjack! Player wins!", true);
            }
            return true;
        }
        return false;
    }

    async dealInitialCards() {
        const deckRect = document.querySelector('.deck').getBoundingClientRect();
        const dealerRect = this.dealerArea.getBoundingClientRect();
        const playerRect = this.playerArea.getBoundingClientRect();

        const dealCard = async (toDealer, faceDown = false, delay) => {
            const card = this.deck.deal();
            const cards = toDealer ? this.dealerCards : this.playerCards;
            const area = toDealer ? this.dealerArea : this.playerArea;
            const areaRect = toDealer ? dealerRect : playerRect;

            cards.push(card);
            area.appendChild(card.element);

            const x = areaRect.left - deckRect.left + (cards.length * 60 - 30);
            const y = areaRect.top - deckRect.top;

            card.faceDown = faceDown;
            card.animateDeal(x, y, delay);

            await new Promise(resolve => setTimeout(resolve, 500));
        };

        await dealCard(false, false, 0);     // Player card 1
        await dealCard(true, false, 500);    // Dealer card 1
        await dealCard(false, false, 1000);  // Player card 2
        await dealCard(true, true, 1500);    // Dealer card 2 (face down)

        setTimeout(() => {
            this.updateScores();
            this.checkForBlackjack();
        }, 2000);
    }

    get playerBalance() {
        return Object.entries(this.playerChips).reduce((total, [value, count]) => total + value * count, 0);
    }

    updateChipCount(value, change) {
        this.playerChips[value] += change;
        if (this.playerChips[value] < 0) this.playerChips[value] = 0;
        this.updateBalanceDisplay();
    }

    setupButtons() {
        document.getElementById('deal-button').addEventListener('click', () => this.startNewGame());
        document.getElementById('hit-button').addEventListener('click', () => this.playerHit());
        document.getElementById('stay-button').addEventListener('click', () => this.playerStay());
    }

    playerHit() {
        if (this.gameState !== 'playing') return;

        const card = this.deck.deal();
        this.playerCards.push(card);
        this.playerArea.appendChild(card.element);

        const x = this.playerCards.length * 60 - 30;
        card.animateDeal(x, 0);
        setTimeout(() => card.flip(), 500);

        setTimeout(() => {
            this.updateScores();
            if (this.calculateScore(this.playerCards) > 21) {
                this.endGame('Bust! Dealer wins!');
            }
        }, 500);
    }

    async playerStay() {
        if (this.gameState !== 'playing') return;

        this.gameState = 'dealer-turn';
        document.getElementById('hit-button').disabled = true;
        document.getElementById('stay-button').disabled = true;

        // Flip dealer's hidden card
        this.dealerCards[1].flip();
        this.updateScores();

        await new Promise(resolve => setTimeout(resolve, 1000));

        // Dealer's turn
        while (this.calculateScore(this.dealerCards) < 17) {
            const card = this.deck.deal();
            this.dealerCards.push(card);
            this.dealerArea.appendChild(card.element);

            const x = this.dealerCards.length * 60 - 30;
            card.animateDeal(x, 0);
            setTimeout(() => card.flip(), 500);

            await new Promise(resolve => setTimeout(resolve, 1000));
            this.updateScores();
        }

        this.determineWinner();
    }

    calculateScore(cards) {
        let score = 0;
        let aces = 0;

        for (let card of cards) {
            if (!card.faceDown) {
                const value = card.getGameValue();
                if (value === 11) aces++;
                score += value;
            }
        }

        // Adjust for aces
        while (score > 21 && aces > 0) {
            score -= 10;
            aces--;
        }

        return score;
    }

    updateScores() {
        const playerScore = this.calculateScore(this.playerCards);
        const dealerScore = this.calculateScore(this.dealerCards);

        this.playerScore.textContent = `Player: ${playerScore}`;
        this.dealerScore.textContent = `Dealer: ${dealerScore}`;
    }

    setupChipDragging() {
        const chips = document.querySelectorAll('.chip');
        const table = document.querySelector('.table');
        const bettingArea = document.querySelector('.betting-area');

        chips.forEach(chip => {
            chip.addEventListener('dragstart', e => {
                if (this.gameState !== 'betting') return;
                chip.classList.add('dragging');
                e.dataTransfer.setData('text/plain', chip.dataset.value);
                e.dataTransfer.setData('application/x-coordinates',
                    JSON.stringify({x: e.offsetX, y: e.offsetY}));
            });

            chip.addEventListener('dragend', () => {
                chip.classList.remove('dragging');
            });

            chip.setAttribute('draggable', 'true');
        });

        table.addEventListener('dragover', e => {
            if (e.target.closest('.deck-area') ||
                e.target.closest('.dealer-area') ||
                e.target.closest('.player-area') ||
                e.target.closest('.controls') ||
                e.target.closest('.chip-rack')) {
                return;
            }
            e.preventDefault();
        });

        table.addEventListener('drop', e => {
            if (e.target.closest('.deck-area') ||
                e.target.closest('.dealer-area') ||
                e.target.closest('.player-area') ||
                e.target.closest('.controls') ||
                e.target.closest('.chip-rack')) {
                return;
            }
            e.preventDefault();
            if (this.gameState !== 'betting') return;

            const chipValue = parseInt(e.dataTransfer.getData('text/plain'));
            const coords = JSON.parse(e.dataTransfer.getData('application/x-coordinates'));
            if (this.playerBalance >= chipValue) {
                this.placeBet(chipValue, e.clientX - coords.x, e.clientY - coords.y);
            }
        });
    }

    placeBet(value, x, y) {
        if (this.playerChips[value] > 0) {
            this.updateChipCount(value, -1);
            this.currentBet += parseInt(value);

            // Create and position new chip at drop coordinates
            const chip = document.createElement('div');
            chip.className = 'chip bet-chip';
            chip.dataset.value = value;
            chip.textContent = '$' + value;
            chip.style.left = x + 'px';
            chip.style.top = y + 'px';
            document.querySelector('.betting-area').appendChild(chip);

            // Enable deal button if there's a bet
            document.getElementById('deal-button').disabled = false;
        }
    }

    clearBettingArea() {
        const bettingArea = document.querySelector('.betting-area');
        const chips = bettingArea.querySelectorAll('.chip');
        chips.forEach(chip => {
            const value = parseInt(chip.dataset.value);
            this.updateChipCount(value, 1);
        });
        bettingArea.innerHTML = '';
        this.currentBet = 0;
    }

    updateBalanceDisplay() {
        const balanceDisplay = document.querySelector('.balance-display');
        balanceDisplay.innerHTML = 'Balance: ';
        Object.entries(this.playerChips).forEach(([value, count]) => {
            if (count > 0) {
                const chip = document.createElement('span');
                chip.className = 'balance-chip';
                chip.textContent = `$${value}:${count} `;
                balanceDisplay.appendChild(chip);
            }
        });
    }

    payoutWinnings(amount) {
        const chipValues = [500, 100, 25, 5];
        for (let value of chipValues) {
            while (amount >= value) {
                this.updateChipCount(value, 1);
                amount -= value;
            }
        }
    }

    startNewGame() {
        if (this.currentBet === 0) return;

        this.gameState = 'playing';
        this.playerCards = [];
        this.dealerCards = [];
        this.dealerArea.innerHTML = '';
        this.playerArea.innerHTML = '';
        this.message.classList.remove('visible');

        document.getElementById('deal-button').disabled = true;
        document.getElementById('hit-button').disabled = false;
        document.getElementById('stay-button').disabled = false;

        this.dealInitialCards();
    }
}

// Start the game when the page loads
window.addEventListener('DOMContentLoaded', () => {
    new BlackjackGame();
});