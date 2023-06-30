"""
Line format for yahoo events:
1241160900 109513 0 |user 2:0.000012 3:0.000000 4:0.000006 5:0.000023 6:0.999958 1:1.000000 |109498 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 |109509 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 [[...more article features omitted...]] |109453 2:0.421669 3:0.000011 4:0.010902 5:0.309585 6:0.257833 1:1.000000

Some log files contain rows with erroneous data.

After the first 10 columns are the articles and their features.
Each article has 7 columns (articleid + 6 features)
Therefore number_of_columns-10 % 7 = 0
"""

import numpy as np 
import fileinput


def get_yahoo_events(filename):
    """
    Reads a stream of events from the list of given files.
    
    Parameters
    ----------
    filenames : list
        List of filenames
    
    Stores
    -------    
    articles : [article_ids]
    features : [[article_1_features] .. [article_n_features]]
    events : [
                 0 : displayed_article_index (relative to the pool),
                 1 : user_click,
                 2 : [user_features],
                 3 : [pool_indexes]
             ]
    """
    global articles, features, events, n_arms, n_events, skiped_articles, fea
    articles = [] #contain id of articles
    features = [] #contain features of articles
    events = [] #contain event data
    skiped_articles = [] #contain articles have non-valid format
    fea = []
    skiped = 0 #the number of event non-valid


    # with fileinput.input(files = filename) as f:
    #     for line in f:
    #         cols = line.split(' ')
    #         if ((len(cols) - 10) % 7 != 0):
    #             skiped += 1
    #         else:
    #             pool_idx = []
    #             pool_ids = []
    #             for i in range(10, len(cols)-6, 7):
    #                 id = cols[i][1:]
    #                 if id not in articles:
    #                     articles.append(id)
    #                     features.append([float(x[2:]) for x in cols[i+1 : i+7]])
    #                 pool_idx.append(articles.index(id))
    #                 pool_ids.append(id)
    #             events.append(
    #                 [
    #                     pool_ids.index(cols[1]),
    #                     int(cols[2]),
    #                     [float(x[2:]) for x in cols[4:10]],
    #                     pool_idx,
    #                 ]
    #             )
    with fileinput.input(files = filename) as f:
        for line in f:
            cols = line.split('|')
            n_cols = len(cols)
            # print(n_cols)
            pool_idx = []
            pool_ids = []
            for i in range(2, n_cols):
                spare_col = cols[i].split(' ')
                if len(spare_col) == 8:
                    # print(1)
                    id = spare_col[0]
                    if id not in articles:
                        articles.append(id)
                        features.append([float(x[2:]) for x in spare_col[1:7]])
                    pool_idx.append(articles.index(id))
                    pool_ids.append(id)
                elif len(spare_col) == 7:
                    id = spare_col[0]
                    spare_col[-1] = spare_col[-1][:-1]
                    if id not in articles:
                        articles.append(id)
                        features.append([float(x[2:]) for x in spare_col[1:]])
                    pool_idx.append(articles.index(id))
                    pool_ids.append(id)
                else:
                    skiped_articles.append(spare_col[0])
                    fea.append(spare_col)
            if (cols[0].split(' ')[1] in pool_ids):
                events.append(
                    [
                        pool_ids.index(cols[0].split(' ')[1]),
                        int(cols[0].split(' ')[2]),
                        [float(x[2:]) for x in cols[2].split(' ')[1:7]],
                        pool_idx,
                    ]
                )
            else:
                skiped += 1
    features = np.array(features)
    n_arms = len(articles)
    n_events = len(events)

    print(n_events, "events with", n_arms, "articles")
    if skiped != 0 :
        print("Skipped articles:", skiped)

# def max_articles(n_articles):
#     """
#     Reduces the number of articles to the threshold provided.
#     Therefore the number of events will also be reduced.

#     Parameters
#     ----------
#     n_articles : number
#         number of max articles after reduction
#     """

#     global articles, features, events, n_arms, n_events
#     assert n_articles < n_arms
 
#     n_arms = n_articles
#     articles = articles[:n_articles]
#     features = features[:n_articles]

#     for i in reversed(range(len(events))):
#         displayed_pool_idx = events[i][0]
#         displayed_article_idx = events[i][3][displayed_pool_idx]
        
#         if displayed_article_idx < n_arms:
#             events[i][0] = displayed_article_idx
#             events[i][3] = np.arange(0, n_arms)
#         else:
#             del events[i]

#     n_events = len(events)
#     print("Number of events:", n_events)
    