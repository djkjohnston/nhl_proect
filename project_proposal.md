### NHL Project
#### Project Proposal
#### March 30th, 2016
###### Dan Johnston
___
Professional sports has seen a rise in professional analytics to supplement coaching and scouting decisions. The NHL has been slower to adopt these methods than other professional sports leagues (MLB leading the pack.)

I would like to explore the following aspects of the National Hockey League
	
1. Can players be segmented into typologies base on individual season performance statistics
2. If a teams mix of player typologies predicts a team's success
3. Reach goal: Look at game level player stats to see if typologies can be predicted early in the season (i.e. by game 10)
    
___
#### Data to be used

The NHL does offer an [API](http://statsapi.web.nhl.com/api/v1/game/2015020743/feed/live), but does not offer any documentation for the API. Looking at some of the raw data (JSON format), it appears to be a play by play feed for each game. My preference is to use this API, but the data may be difficult to work with. If it becomes too cumbersome, I will resort to web scraping.

In addition to the API, the NHL maintains a number of tradition HTML pages that contain relevant stats. A few examples:
* [Game Summary](http://www.nhl.com/scores/htmlreports/20142015/GS021217.HTM)
* [Player statistics by game](http://www.nhl.com/scores/htmlreports/20142015/ES021217.HTM) or [Alternate](https://www.nhl.com/gamecenter/nyr-vs-wsh/2015/04/11/2014021217#game=2014021217,game_state=final,game_tab=boxscore)
* [Player statistics by season](http://www.nhl.com/stats/player?reportType=season&report=skatersummary&season=20142015&gameType=2&sort=points&aggregate=0&teamId=15&pos=S)

[Nice Time On Ice](http://www.nicetimeonice.com/api) offers a RESTful API which can be used to gather all of the season, team, player, and game IDs which are used by the NHL website. This will make either using the API or manipulating URLs for web scraping significantly easier.