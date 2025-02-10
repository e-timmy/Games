class Chip {
    constructor(value) {
        this.value = value;
        this.element = this.createChipElement();
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.setupDragListeners();
    }

    createChipElement() {
        const chip = document.createElement('div');
        chip.className = `chip chip-${this.value}`;
        chip.textContent = this.value;
        return chip;
    }

    setupDragListeners() {
        this.element.addEventListener('mousedown', this.startDragging.bind(this));
        document.addEventListener('mousemove', this.drag.bind(this));
        document.addEventListener('mouseup', this.stopDragging.bind(this));
    }

    startDragging(e) {
        this.isDragging = true;
        const rect = this.element.getBoundingClientRect();
        this.dragOffset.x = e.clientX - rect.left;
        this.dragOffset.y = e.clientY - rect.top;
        this.element.style.zIndex = '1000';
    }

    drag(e) {
        if (!this.isDragging) return;
        const x = e.clientX - this.dragOffset.x;
        const y = e.clientY - this.dragOffset.y;
        this.element.style.left = `${x}px`;
        this.element.style.top = `${y}px`;
    }

    stopDragging() {
        this.isDragging = false;
        this.element.style.zIndex = '';
    }

    moveTo(x, y) {
        this.element.style.left = `${x}px`;
        this.element.style.top = `${y}px`;
    }
}

class ChipManager {
    constructor(playerChipsElement, potElement) {
        this.playerChipsElement = playerChipsElement;
        this.potElement = potElement;
        this.playerChips = [];
        this.potChips = [];
        this.totalValue = 1000; // Starting chip value
    }

    createInitialChips() {
        const chipValues = [5, 10, 25, 50, 100];
        chipValues.forEach(value => {
            const chip = new Chip(value);
            this.playerChips.push(chip);
            this.playerChipsElement.appendChild(chip.element);
            this.arrangePlayerChips();
        });
        this.updateChipTotal();
    }

    arrangePlayerChips() {
        this.playerChips.forEach((chip, index) => {
            const x = index * 60;
            const y = 0;
            chip.moveTo(x, y);
        });
    }

    updateChipTotal() {
        const totalElement = document.querySelector('.chip-total') || document.createElement('div');
        totalElement.className = 'chip-total';
        totalElement.textContent = `Total: $${this.totalValue}`;
        this.playerChipsElement.appendChild(totalElement);
    }

    addChipToPot(chip) {
        this.potChips.push(chip);
        this.potElement.appendChild(chip.element);
        this.arrangePoChips();
        this.totalValue -= chip.value;
        this.updateChipTotal();
    }

    arrangePoChips() {
        this.potChips.forEach((chip, index) => {
            const x = 75 + Math.random() * 50;
            const y = 25 + Math.random() * 50;
            chip.moveTo(x, y);
        });
    }

    clearPot() {
        this.potChips.forEach(chip => {
            chip.element.remove();
        });
        this.potChips = [];
    }

    getPotValue() {
        return this.potChips.reduce((sum, chip) => sum + chip.value, 0);
    }
}
