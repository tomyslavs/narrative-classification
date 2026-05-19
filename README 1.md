# News Portals Dataset

This dataset contains normalized CSV files with article metadata and
user comments scraped from Lithuanian news portals.

**Sources:** Alfa, Delfi, Respublika, 15min, TV3

------------------------------------------------------------------------

## File Overview

-   `articles.csv` --- article metadata and content\
-   `comments.csv` --- user comments linked to articles via `article_id`

------------------------------------------------------------------------

## articles.csv

**Description**\
Article metadata and content. Each row represents one scraped article.
Articles are uniquely identified by `article_id`.

**CSV format** - Delimiter: comma (`,`) - Encoding: UTF-8

### Example entry

  ----------------------------------------------------------------------------------------
  Field                               Example value
  ----------------------------------- ----------------------------------------------------
  article_id                          2593408

  url                                 https://www.15min.lt/naujiena/aktualu/pasaulis/...

  title                               Rusija sako, kad per ukrainiečių dronų ataką žuvo
                                      mažiausiai 20 žmonių

  published_at                        2026-01-01T14:21:02+02:00

  category                            pasaulis

  keywords                            \["Rusija","Ukraina","Dronas"\]

  comments_count                      1

  article_content                     Trys bepiločiai orlaiviai smogė kavinei ir
                                      viešbučiui...
  ----------------------------------------------------------------------------------------

### Field descriptions

  -----------------------------------------------------------------------
  Field                   Type                    Description
  ----------------------- ----------------------- -----------------------
  article_id              string                  Unique identifier of
                                                  the article within the
                                                  portal. Used to link
                                                  articles.csv to
                                                  comments.csv

  url                     string                  URL of the article page

  title                   string                  Article headline text

  published_at            string (datetime)       Article publication
                                                  timestamp

  category                string                  Article category label

  keywords                string (JSON array)     Article keywords

  comments_count          int                     Number of total
                                                  comments for the
                                                  article

  article_content         string                  Article body text
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## comments.csv

**Description**\
User comments under articles. Each row represents a single comment
associated with an article via `article_id`.

**CSV format** - Delimiter: comma (`,`) - Encoding: UTF-8

### Example entry

  Field               Example value
  ------------------- ------------------------------------------------------------
  article_id          2593408
  comment_id          69563d41c52ab2593340
  username            Algis Sėliškis
  comment             Ir turtuoliai priešgaisrinei signalizacijai pinigų neturi?
  created_at          2026-01-01 11:24:17
  is_reply            false
  parent_ref          (empty)
  reactions_like      1
  reactions_dislike   0

### Field descriptions

  -----------------------------------------------------------------------
  Field                   Type                    Description
  ----------------------- ----------------------- -----------------------
  article_id              string                  Identifier of the
                                                  article this comment
                                                  belongs to

  comment_id              string                  Unique identifier of
                                                  the comment within the
                                                  website

  username                string                  Username of the comment
                                                  author

  comment                 string                  Comment text content

  created_at              string (datetime)       Comment publication
                                                  timestamp

  is_reply                boolean                 Indicates whether the
                                                  comment is a reply

  parent_ref              string / null           Parent comment
                                                  reference; threads can
                                                  be reconstructed
                                                  recursively

  reactions\_\*           int                     Flattened reaction
                                                  count for a specific
                                                  reaction type
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Reactions by website

  Portal       Reaction types
  ------------ ---------------------------------------------------
  15min        like, dislike
  Alfa         like, dislike
  Respublika   like, dislike
  TV3          like, love, care, laugh, surprised, sad, angry
  Delfi        like, dislike, laugh, love, sad, angry, surprised
