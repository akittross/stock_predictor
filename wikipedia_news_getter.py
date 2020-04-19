

import requests


from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def month_getter(month, year):
    url = "https://en.wikipedia.org/wiki/Portal:Current_events/" + month + "_" + year
    resp = requests.get(url)
    return strip_tags(resp.text)

def last_day_of(month):
    if month == 'February':
        return 28  #fuck leap years
    elif month == 'September':
        return 30
    elif month == 'April':
        return 30
    elif month == 'June':
        return 30
    elif month == 'November':
        return 30
    else:
        return 31

def clean_text(text):
    words = text.split()
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    lower = ''
    for word in stripped:
        lower += word.lower() + ' '
    return lower

def get_and_write_days(month, month_text, year):
    tag = 'Current events of ' + month
    day_list = month_text.split(tag)
    day_number = 0
    for day_text in day_list:
        if day_number > 0 and day_number <= last_day_of(month):
            filename = "wpnews/" + str(day_number) + '_' + month + '_' + year + ".txt"
            f = open(filename, "w")
            f.write(clean_text(day_text))
            f.close()
        day_number += 1

def main():
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    year = 2008
    while year < 2020:
        for month in months:
            month_text = month_getter(month, str(year))
            get_and_write_days(str(month), month_text, str(year))
        year += 1



if __name__ == "__main__":
    main()

