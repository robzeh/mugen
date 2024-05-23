import click
import pandas as pd
from rich.console import Console
console = Console()

@click.command()
@click.option('--csv')
def main(csv):
    df = pd.read_csv(csv)
    console.print(df.columns)


if __name__ == '__main__':
    main()