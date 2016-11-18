#!/usr/lib/python3

from dateutil import relativedelta
from datetime import date


def compute_age(row):
    """
        Given a player, function returns the years of player.
        row: Row of the DataFrame, representing a dyad which contains a player.
    """
    data_date = date(2013, 1, 1)
    delta = relativedelta.relativedelta(data_date, row['birthday'])

    return delta.years


def pondered_number_of_cards(row, cards_name):
    """
    Given a player, function analyzes the skin color and balance the number of received cards if player is black.

    row: Row of the DataFrame, representing a dyad which contains a player.
    cardsName: Type of received cards for the player
    """
    nb_cards = row[cards_name]

    if row['associationScore'] > 0:
        coef = (row['rater'] / 100) * row['associationScore']
    elif row['associationScore'] < 0:
        coef = (1 - (row['rater'] / 100)) * row['associationScore']
    else:
        coef = 0

    nb_cards += nb_cards * coef

    return nb_cards

