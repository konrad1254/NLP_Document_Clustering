"""
extraction.py
~~~~~~~~~~~~~
The extraction method uses the wikipedia API to generate a random search. If there multiple entries for one search, a new random search gets started
"""

# Import
import wikipedia as w
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Wikipedia extraction
def wiki_extraction(n, languages):
      for lan in languages:
            if lan not in w.languages():
                  raise Exception('Language wrongly specified')
            if isinstance(lan, str) is False:
                  raise Exception('Wrong format. The language list takes only strings')
      rows_list = []
      for lan in languages:
            w.set_lang(lan)
            page = w.random(pages = n)
            for i in page:
                  try: 
                        dict_ = {}      
                        summary = w.WikipediaPage(title=i).summary
                        if len(summary) > 0:
                              input_dict = {'Title': i, 'Text': summary, 'Language':lan}
                              dict_.update(input_dict)
                              rows_list.append(dict_)
                        else:
                              page.append(w.random(pages = 1))
                  except:
                        page.append(w.random(pages = 1))
                        pass  
      return pd.DataFrame(rows_list)
