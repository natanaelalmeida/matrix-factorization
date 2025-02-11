import pandas as pd
import numpy as np

users = ["Ripley", "Darth Vader", "Spock", "Hermione"]

musics = [
    {"title": "Another One Bites the Dust", "genre": "Rock"},
    {"title": "Survivor", "genre": "Pop"},
    {"title": "Don't Stop Believin'", "genre": "Rock"},
    {"title": "The Imperial March", "genre": "Soundtrack"},
    {"title": "O Fortuna", "genre": "Opera"},
    {"title": "Back in Black", "genre": "Heavy Metal"},
    {"title": "Clair de Lune", "genre": "Instrumental"},
    {"title": "Comfortably Numb", "genre": "Rock"},
    {"title": "Brandenburg Concerto No. 3", "genre": "Baroque"},
    {"title": "Hedwig’s Theme", "genre": "Soundtrack"},
    {"title": "Bohemian Rhapsody", "genre": "Rock"},
    {"title": "Firework", "genre": "Pop"}
]

user_history = [
    {"user": "Ripley", "music": "Another One Bites the Dust", "rating": 5},
    {"user": "Ripley", "music": "Survivor", "rating": 5},
    {"user": "Ripley", "music": "Don't Stop Believin'", "rating": 4},
    {"user": "Ripley", "music": "The Imperial March", "rating": 4},
    {"user": "Ripley", "music": "O Fortuna", "rating": 5},
    {"user": "Ripley", "music": "Back in Black", "rating": 5},
    {"user": "Ripley", "music": "Clair de Lune", "rating": 2},
    {"user": "Ripley", "music": "Comfortably Numb", "rating": 3},
    {"user": "Ripley", "music": "Brandenburg Concerto No. 3", "rating": 1},
    {"user": "Ripley", "music": "Hedwig’s Theme", "rating": 0},
    {"user": "Ripley", "music": "Bohemian Rhapsody", "rating": 2},
    {"user": "Ripley", "music": "Firework", "rating": 4},
    {"user": "Darth Vader", "music": "Another One Bites the Dust", "rating": 4},
    {"user": "Darth Vader", "music": "Survivor", "rating": 0},
    {"user": "Darth Vader", "music": "Don't Stop Believin'", "rating": 2},
    {"user": "Darth Vader", "music": "The Imperial March", "rating": 5},
    {"user": "Darth Vader", "music": "O Fortuna", "rating": 4},
    {"user": "Darth Vader", "music": "Back in Black", "rating": 4},
    {"user": "Darth Vader", "music": "Clair de Lune", "rating": 1},
    {"user": "Darth Vader", "music": "Comfortably Numb", "rating": 2},
    {"user": "Darth Vader", "music": "Brandenburg Concerto No. 3", "rating": 0},
    {"user": "Darth Vader", "music": "Hedwig’s Theme", "rating": 0},
    {"user": "Darth Vader", "music": "Bohemian Rhapsody", "rating": 2},
    {"user": "Darth Vader", "music": "Firework", "rating": 0},
    {"user": "Spock", "music": "Another One Bites the Dust", "rating": 2},
    {"user": "Spock", "music": "Survivor", "rating": 2},
    {"user": "Spock", "music": "Don't Stop Believin'", "rating": 3},
    {"user": "Spock", "music": "The Imperial March", "rating": 0},
    {"user": "Spock", "music": "O Fortuna", "rating": 4},
    {"user": "Spock", "music": "Back in Black", "rating": 0},
    {"user": "Spock", "music": "Clair de Lune", "rating": 5},
    {"user": "Spock", "music": "Comfortably Numb", "rating": 4},
    {"user": "Spock", "music": "Brandenburg Concerto No. 3", "rating": 5},
    {"user": "Spock", "music": "Hedwig’s Theme", "rating": 3},
    {"user": "Spock", "music": "Bohemian Rhapsody", "rating": 5},
    {"user": "Spock", "music": "Firework", "rating": 2},
    {"user": "Hermione", "music": "Another One Bites the Dust", "rating": 1},
    {"user": "Hermione", "music": "Survivor", "rating": 2},
    {"user": "Hermione", "music": "Don't Stop Believin'", "rating": 2},
    {"user": "Hermione", "music": "The Imperial March", "rating": 1},
    {"user": "Hermione", "music": "O Fortuna", "rating": 2},
    {"user": "Hermione", "music": "Back in Black", "rating": 0},
    {"user": "Hermione", "music": "Clair de Lune", "rating": 5},
    {"user": "Hermione", "music": "Comfortably Numb", "rating": 2},
    {"user": "Hermione", "music": "Brandenburg Concerto No. 3", "rating": 0},
    {"user": "Hermione", "music": "Hedwig’s Theme", "rating": 5},
    {"user": "Hermione", "music": "Bohemian Rhapsody", "rating": 5},
    {"user": "Hermione", "music": "Firework", "rating": 5}
]

def get_user_history() -> pd.DataFrame:
    return pd.DataFrame(user_history)

def get_matrix_interaction() -> pd.DataFrame:
    df_history = pd.DataFrame(user_history)
    return df_history.pivot_table(values=['rating'], columns=['music'], index=['user'])