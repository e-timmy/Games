function animateCard(cardElement, startX, startY, endX, endY, duration, onComplete) {
    cardElement.style.transition = 'none';
    cardElement.style.transform = `translate(${startX}px, ${startY}px)`;
    cardElement.style.opacity = '1';

    setTimeout(() => {
        cardElement.style.transition = `transform ${duration}ms ease-out`;
        cardElement.style.transform = `translate(${endX}px, ${endY}px)`;

        setTimeout(() => {
            if (onComplete) onComplete();
        }, duration);
    }, 50);
}

function flipCard(cardElement, duration) {
    return new Promise((resolve) => {
        cardElement.style.transition = `transform ${duration}ms`;
        cardElement.style.transform = 'rotateY(90deg)';

        setTimeout(() => {
            cardElement.classList.remove('face-down');
            cardElement.textContent = cardElement.dataset.cardValue;
            cardElement.style.transform = 'rotateY(0deg)';
            setTimeout(resolve, duration / 2);
        }, duration / 2);
    });
}