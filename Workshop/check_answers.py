
from termcolor import colored
from IPython.display import HTML
import matplotlib.pyplot as plt
import time

class check_answers():
    def __init__(self, x):
        self.x = x

    def check_load(*args):
        for i in args:
            assert len(i) >= 10000, colored('Da hat wohl was nicht geklappt',
                                            'red')
            if len(i) >= 10000:
                print(colored('Das hat geklappt!', color='green'))
            else:
                pass

    def check_col(x):
        assert x == 8, colored('Das war wohl nicht richtig', 'red')
        if x == 8:
            print(
                colored(f'{x} ist die richtige Anzahl an Spalten!',
                        color='green'))

    def check_drp_col(x):
        y = x.shape[1]
        assert y == 8, colored(
            'Hast du vielleicht den "inplace"-Parameter vergessen',
            color='red')
        if y == 8:
            print(colored('Du hast alles bedacht!', color='green'))

    def check_unnamend(*args):
        for i in args:
            y = i.columns.tolist()
            if 'Unnamend: 0' in y:
                print(
                    colored('Die unnötige Spalte ist noch vorhanden',
                            color='red'))
            else:
                print(colored('"Unnamend: 0" wurde entfernt!', color='green'))

    def check_mean(x,df):
        assert x==round(df['Anzahl_Mieteinheiten'].mean()), colored("Versuche es mal mit df_kundenstamm['Anzahl_Mieteinheiten'].mean()", color='red')
        if x == round(df['Anzahl_Mieteinheiten'].mean()):
            print(colored('Richtig! Die durchschnittliche Anzahl an Mieteinheiten beträgt {}'\
                          .format(round(df['Anzahl_Mieteinheiten'].mean())),
                          color='green'))

    def check_typ_str(*args):
        for i in args:
            if i.dtype == 'object':
                print(
                    colored('Jetzt ist die Spalte {} ein Objekt'.format(
                        i.name),
                            color='green'))
            else:
                print(
                    colored(
                        'Da stimmt was nicht, überrprüfe bitte nochmal deinen Code',
                        color='red'))

    def check_dupl(*args):
        for i in args:
            assert i.duplicated().sum() == 0, colored(
                'Es sind noch duplikate enthalten', color='red')

            if i.duplicated().sum() == 0:
                print(colored('Duplikate wurden entfernt', color='green'))

    def check_shape(x):
        assert x == 'df_beschwerden' or x == 'beschwerden' or x == 'df_b', colored(
            'Schau dir mal den df_beschwerden an', color='red')
        if x == 'df_beschwerden' or x == 'beschwerden' or x == 'df_b':
            print(
                colored('Genau, denn hier ist die Beschwerde-ID federführend',
                        color='green'))

    def check_grp(x):
        assert len(x) <= 4000, colored(
            'Überprüfe nochmal deinen Code, da hat was nicht geklappt',
            color='red')
        if len(x) <= 4000:
            print(
                colored('Du hast den Datensatz erfolgreich gruppiert',
                        color='green'))

    def check_join(x):
        y = x.shape[1]
        assert y == 13, colored(
            'Der Datensatz sollte 13 Spalten umfassen', color='red')
        if y == 13:
            print(
                colored('Sehr gut! Alle Daten wurden erfolgreich gejoint',
                        color='green'))

    def check_na(x):
        assert x.isna().sum().sum() == 0, colored(
            'Es sind noch fehlende Werte vorhanden', color='red')
        if x.isna().sum().sum() == 0:
            print(
                colored('Sehr gut!Du hast die Null imputiert!', color='green'))

    def check_mldf(x):
        assert x.shape[0] == 10000 and x.shape[1] == 8, colored(
            'Überpüfe nochmal die Feature-Auswahl', color='red')
        if x.shape[0] == 10000 and x.shape[1] == 8:
            print(colored('Sehr gut! Es hat alles geklappt', color='green'))
    
    def check_ml_dummy(x):
        assert x.shape[0] == 10000 and x.shape[1] == 16, colored(
            'Überpüfe nochmal das Dummyisieren', color='red')
        if x.shape[0] == 10000 and x.shape[1] == 16:
            print(colored('Sehr gut! Es hat alles geklappt', color='green'))
    
    def check_X_Y(x,y):
        assert (x.shape[0] == 10000 and x.shape[1] == 15) and (y.shape[0] == 10000), colored(
            'Da ist irgendwo ne Spalte zu viel oder zu wenig :-)', color='red')
        if (x.shape[0] == 10000 and x.shape[1] == 15) and (y.shape[0] == 10000):
            print(colored('Yeah! Einen Schritt weiter', color='green'))
    
    def check_train_test(x,y):
        assert (x.shape[0]==7500 and y.shape[0]==2500), colored('Hast du den test_split auf 0.25 gesetzt?!',
                                                               color='red')
        if (x.shape[0]==7500 and y.shape[0]==2500):
            print(colored('Sehr gut! Du hast den Train-Test-Split durchgeführt!',
                                                               color='green'))    