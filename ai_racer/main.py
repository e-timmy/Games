import argparse
import pygame
from game import Game
from menu import Menu, PlayerConfig


def get_headless_config():
    print("\n=== Headless Racing Configuration ===")

    # Get number of AI players
    while True:
        try:
            num_players = int(input("Enter number of AI players (1-10): "))
            if 1 <= num_players <= 10:
                break
            print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")

    # Get number of laps
    while True:
        try:
            target_laps = int(input("Enter number of laps (0 for infinite): "))
            if target_laps >= 0:
                break
            print("Please enter a non-negative number.")
        except ValueError:
            print("Please enter a valid number.")

    # Get simulation speed
    while True:
        try:
            speed_factor = int(input("Enter simulation speed (1-100): "))
            if 1 <= speed_factor <= 100:
                break
            print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Please enter a valid number.")

    print("\nStarting simulation...")
    return num_players, target_laps, speed_factor


def main():
    parser = argparse.ArgumentParser(description='Racing Game')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    args = parser.parse_args()

    headless = input("Headless (y/n): ")
    args.headless = True if headless == 'y' else False

    if not args.headless:
        pygame.init()
        screen = pygame.display.set_mode((1024, 768))
        pygame.display.set_caption("Racing Game")
    else:
        screen = None

    while True:
        if args.headless:
            num_players, target_laps, speed_factor = get_headless_config()
            player_configs = [PlayerConfig("AI", None, 1.0) for _ in range(num_players)]
        else:
            menu = Menu(screen)
            player_configs, target_laps = menu.run()
            speed_factor = 1

        if not player_configs:
            break

        game = Game(screen, player_configs, target_laps, headless=args.headless, speed_factor=speed_factor)

        running = True
        while running:
            if not args.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        return

            finished = game.update()

            if finished:
                if args.headless:
                    print("\nRace finished!")
                    winner = game.winner
                    print(f"Winner: Car {winner.number}")
                    print(f"Laps completed: {winner.laps}")

                    retry = input("\nRun another race? (y/n): ").lower()
                    if retry != 'y':
                        return
                    break
                else:
                    break

            if not args.headless:
                game.draw()
                pygame.display.flip()

    if not args.headless:
        pygame.quit()


if __name__ == "__main__":
    main()