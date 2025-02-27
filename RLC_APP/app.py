import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kaggle
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv(r"C:\Users\James Wilson\OneDrive\Documents\Work\Coding\Jupyter\Lessons\RL-Champ\Code\data\cleaned_RLC_data.csv")
# Set page title and icon
st.set_page_config(page_title="Game Class Dataset Explorer", page_icon= '🎮' )

#Step 1- Sidebar navigation

page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis","Model Training and Evaluation", "About"])

#Step 2- Adding Content to Pages

if page == "Home":
    st.title("🚗 Global Video Game Sales Dataset Explorer")
    st.subheader('Enjoy this video game sales dataset explorer app')
    st.write("""
    This app provides a platform for exploring some data on Rocket League Championships 2021-2022 and the stats of Profesional teams.
    Feel free to look through the visual representations provided in the app.
    Use the side bar to navigate through the apps pages.
    """)
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTfJQM2PKoCN1NTkNRHaXHqWTQbg6cNHSTZxQ&s", caption='The Starter Car(Octane)')

elif page == "Data Overview":
    st.title("⚽ Data Overview")
    st.subheader("About the Data")
    st.write("""
    Rocket League is a very popular game that mixes soccer with rocket powered vehichels. There is both casual and ranked modes as well as tournaments allowing players to really test their metal. This data set is of Profesional Teams going head to head. Join me as I do some EDA to show you a few of the stats that can affect a teams winning chances. Then we will do some modeling on the data to predict how often a team wins as opposed to losing.
    """)
    st.image("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMTEhUSExMWFhUXFxUYFhgXGBgVGBgXFxcXFhcWFxUYHSggGBolHRUXITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OGhAQGi0lHSUtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAKgBLAMBIgACEQEDEQH/xAAbAAABBQEBAAAAAAAAAAAAAAAEAAECAwUGB//EAEIQAAEDAQUECAMGBQMDBQAAAAEAAhEDBBIhMUEFUWFxEyKBkaGxwfAyQtEGI1JicuEUM4KS8VOiwpOy0hUkNENj/8QAGQEAAwEBAQAAAAAAAAAAAAAAAAECAwQF/8QAJxEAAgIBBAEEAgMBAAAAAAAAAAECESEDEjFBURMiMnFhgQSR0SP/2gAMAwEAAhEDEQA/APGf4V2kHkQVB1IjMEJEcMUxdvnvQBMMGYJ7k2qgHbk95AhyEwUmhJoGqYibcFBzdysMFQMYpAMAnISpqbpCfYEWmFJRLpTpiLGlWh6HapNKBMPstfEDetNoWKxwBGK2KcLWLIZe1iuYxUteBmURReDkVViJtpq1tGVKm1GUKaZDZVSs59/RaVmsqus1nWrZrHuWcpDQHSsitNlWrSs+9WOoLJyKowKlmQdezroqtBZ9ooqoyEzAqUVQ6mtatTQrqa1TIACxN0aMLFEtTAF6NZ1utBacAeGUE9q2HNWXtSjeIxyzw7lE3gqPJl1zMk5qgVTichp9VNzdIQ9pBEDf70XOlZuhnWlUurO3nvUroSdTKqkh5Gp2l+hJ54ottTeAlY9nVHNNQNdcGZ03Jm0vf+FEpFJMGtGz3BxZdN4GIGKCr0S0wQQVt09puZUFRuDwQQZxkHPFS25YqzotFWIqy69IxgwZA+iUZPsuUV0c9EJiFc6m3eTyHqVFzBx8FqZlbXQVIEpoTgIoC0MHeqXhWXRCTQEkIjTanfKcAyp9GSmBSnaVNzSowmA8KbJUbqsa1AiSNokhuvBCBqupIsVF5eTmj9mGJk6ISnTRlnpwQdOCFIHEtstR0gRK36NpgSWndwnn2LnQ1H2bmrRlJHZ7KIe2YjRb1npLD+zpbdgc8V0tmhZT5CJNtlJGSoqU4XXWN1MU8YXM21wvGFlGVstqjOqBAWhiPrFAVnLWJEmZ9WmhX00fUQzhOIWqMwN9NUOCMe1UOCqxg91Y21mgGACTmSfSVtVXQFh2594kQTvx03ZrPUkuGaQRnUgXHnqdApONN1Vo+WIccscjOGEJVKmctPZ4ckTsVlAvd0xfdumA3O9BgY6THislKkzZRtlu0NmUW1AKT77SGkmIxIkgcjhKltPo3ABjA0AQYxk6nh+yz30w5xh3ITKIAiPAZ4+/RYy+zbBULZUa00wTdMSMgYyQ1SrBgH/KutVculxz3k66e+CBa7fHfPomlYrpDUKhcQZwxvcxzynDJD2q1F5zw0GiWznwbpydh26fRVhpBg6SO5apUTJ2hhxHandSg4nszPar7M3GdQCRz949irLFRBW5g9wmDVcAEmtxTAGcxJrFfdThiGKyq4rKZVjqW5WNF3JJsZF7cMowQzWI09ZUFmMJxE2VFse/2U2s19+andTtamA5HuFfRaoCYRNHkpY0aez7KXQM/FadXZZaJIKG2TaQxwO4g7luW7bAcy7yjsJ781zSlLdg6Yxi4nNvpwrKLwEPaq2KDfaty64PByTidhsvagZ80HdAx7VsUNrlz5mBu4Lzyx1TIK3qdqDQqqznlE63aP2kgQDhhPYhRt4ucOHquOtNq8dFOz13txB0xQopBTO1O0wdOak+u2JkBce60uxM4qipbScCU9oUzqbcJAunBUVKhDbowiB9VhWTaBbM+qvo2y98RGeHKCnQ0g2taw1t52+MEJY7WXgl0Z+Cz7XWvGBkMlWyoQIGSHgpI0bVaxkMSexZD6ZGoIO/6p2jGVS+pOHiuWUnJmsVRC11QRAwPnIiQUAAR6fur7S8YjXPtQjqs4RqFcVgovsrgCXnIefJT/8AUxMQRlGsoWuRAbxk8z78FVRGN6MshxOQScE8stc0F7Qqkw1rTn7CsZDcIGOOM8vRIGTlwz1yjio1ZlZ/gbebMxwxwRVdpdDxr8XMfUKmPBEWV/ynI+x74rVkLwV0+qQY/caqdRsGBlmORxCXAqUy3l5H9/NMkpDdE4CRKZzlfQiZZmnuqNR2IO8BO12fI/VTWA7LB3J6bcN25Uh6tDhGeORHiCEqCybYx3quqzVVgrTsdjDhLjAifRXGDbwJyM9rE4YtYWSkPmd2BTp2SjBJcYHDHsC0elXYk76MrgpsKstlC64huPvNVCm6Mh3hZyg0UmFUqsKb7SYHL1QjaLjAEd6TqboGWuvJTsLtk3mZTCiqzRfOXiE5ov3HxVpUQ7YVRAaZzCIqWlZjahbmESy20/mb5/VUQ0Wh5ceSn0pA1UG2uz8e4pOtVDee45Jiol/FGUxMpzaaMw0k9zfNMbXSGru6UWgodrlJpgzuURaaP+o4f0wotr0hPXPDAotBTLVI4AKnp6ZEB3vuUXVQcA7xUsqJfaIGqFJbnMKQGGMoatTxDffH6LPb4NEV1ng5DGc8VVSbicPfvzRFOl3Y/umY3AwPfue9FdDTAalWSfBTLroAGeZ5n3HeiG2fGSMBifQIVzZJO9DWaKXF+Qyx1i7TIZ+Sk5iVmLQ3HDEe8Mf8KxovYgz75LBrOEVRlOcJUjVETvkdu/17FAtBzOPZ3KdKm0giOOe79pWjkjNIsq1ZAd2Hn+4UKdeCJ+E4Hkc1ZZw3ERgeZ81MUx+D33pbga7BnCCWnl9CptoOjKOeHmtiy0W/GRg2BzPyj3oCo07D0rpDpx99i1UklcgUHJ0jM6HKXDz8lbQoNJAF5x3AIi10BTe2m1t9xE79XRAHBoPaoMtr2Q4m6cw1oA7SRjHnwwUPVVe1BtSeSx+zy3NhHFxACdtmZq6mP6i7yQtotr6hlxnnCpNRT6kwe3pGnRs7HGGvaToA0yf03hieAxVrCGz8RnOWiPPD/CymuGoB5zHgURY4cHQAHy26OGMhk/PN2Nc4xhHqyXYsdINfWjI4aTA8Ev4iTi4TpJ9RgsxzZ071AyNwUbh7n5NlzhGMD810uB7YjuUf4h8QAT+kgd9zHxWbSe4Ygu5iQO05LUoFhDetT6TrXpLRGJAu3eq4RjMk46DEpy8hbBKNsc0zed2HvTG0kmbzu1aUDVzT2sHjJThgIJhpAzOD45gT5KXJIaTZnGu44SD74qynWcM6bTrqPVWWxjOqGkNIHWg3WukyIAy4zd5DUE0HjENw3jrAcy3AKlITbRrUNo0p+8YeTXTj2geaqNrs5zBHNoPiDKyw9+/2RHkmLxqN3DwCbbfZS1GvH9Go5lmdkWdoc3zhUOsTNA0/pd+5QEs3kZ6TyxERrvy44TqUQGg3s8Ph6pOsPB+h4ITa7ZXq3ykEOoN1Dh4+gTssbDkfD/KEFZwEXjAyxPhuVjLY8ZO9fMFO35BTh3EJds/j5+kKqvs93ywe391Ju0na3TzEHwhauzX0HNJqENPB3kDn3pOco5LXpSxwc+bJUbiR70UOleMF0LbpcRTqNPAmCfr2JVKP46fgq9Z9ofoRfxZgMtb9ch7+igK7rxJPNbo2bScCBI8fT3CAtmzTTddMmQCOMmBCa1YsmWlKKt8AbHH379wkXlb9j+z56I1Owamc8RpgMN+KyatlumCMsT5Ae96S1Yt4G9JxSvspe8hsa5ntyHd5qsDCUnPk8z4lM54+iuxYJMbJ4laJf0cNEYDHmqLE2OtvwHqVRVfeJKTdjSBalWmzAfeHfEN7syraNSrUgwYy3DhA71pWTYbRDiJ1jMDlwWkynoBGHj7lYS1F0RTMejs7WefBHWawFzg3MmPojKNDx9/RaTqAYyT8Tgbv5W5E9uI5SoWpnJcNKU8IzbXRbAYzITHE6u8uwBHbJsLabC8jBjXOPYCfQrPp0yagxyH7rbthizuZnfLKf/Ue1h81rrzt0uDXThsi2+TkrbZSazyReFNtJp4ubTYyD2gk8ism0B0kuzK6qltKmx73PaC11ao7A5gPi7mDME5H5hisjbNvoPJuNccsTDecj6fss4yldVg5nGNXeTLYUrpJwBJ3DHyUmVtzWjsvf90qVSq4iC4kbpw7BkFqZjspkZw3mQD/AG5+CkWN/Fj+Vs/9xCFhTY9DAPFsa49dpmCLxMkn5b0XZ3TnG+IU6jHNggtumQ1zBgYzgkXh2wcQgCJUqdVzJukgEQYJEjODGY4KK8DLKkHE4njj5qDnRorzaKb7xLbjsLoZizjIcZbO8GOCubYySAwioSJFyXZZ9UgOB4EIuhAAeVbTznI6EfVW1KJBIIIIzBBBHMaZjvVDyUXYBotJPxQ/9WJ/vEO8VM2hhMy5h3jrD0Lf9yy5UmuKNo7ZrgFwwLap1GbjyDgKncqW2drnFt0scAZ6wERj8NSCTwvIJrRqi6dseABekDIOh4HIOmOyFLxwFrsh/Bt0e08D1D/4/wC5X2XZ9TFwhrMnPcQKZ1uyPj/S2TwWhUsrKQD61MdKRLaILg0A4tfXkyJzFMEE6wgbVaXVDLjMCAMAGj8LWjBreAEKdzNI6dlZo0w+bpc2Mg4ht6TlLb1zLAkHipl7dKdNv9N4977x8VWki2aqCQnNYc2DmCQezGB3KH8O2MC4Y6w4XY3QMZ1lFWKyuqvbTZ8TjAnIalx/KBJPAJ9oWql0hFJgNNsNa43g590QahIMS4ycsJCakyJKK5AjYpye3xafEQO9W0alopjqy4bhFQdt0mFIVaZ0e3uf/wCMeKXRgzD2nPUtP+8AdxKrc+yUo9M68WBpF7L0wkeYWbbgwxIks+A750PI+q1icHQRBunPQsYZHDFZVUfeXRBumeEn4Qe3yWSO6Mvaa+yrUKbYfi0w0jec3GeGnFqu2uCBi1lRpALSWjrNOsjXQjesK01peGtyGGeMzjniug2XahVHQPw/A44AO5n5Tl3cVhqLa9xtCVrazlLTRsr5lvQu3jKThkMD4c1nWz7PFoDm1GvYYggQcd0SD3zwXQ7bsYBIIAIOIOhWJQeKWIEiZuyQJGExv4rq05OrTOXVjG6kv2Z+1HGmQ0tLcIaCIwymeKzb/wCYds+i7W1W26zFgqs+ebt5s/CHsxaTxyPBD09iWeoL7WCDuJbHCNOxaLWpZRL0G37WV2d1x1w5fJy1b2eXJEngI3qqtTvDDMYg8dFbYW3xljkRnBGEe9IWD8mcfAbYbOHEuceq3F3oBxOSVsq3ydNTwA09ErbWDIpA/Di4/m3ch6qrpIEanE+g96qEryehFKEfyR2dQMznOKPtuDrOP/0dUP6aNN7/ADDVOwtx7vBC7eq/eVNOjsdU/wBVUtpjwlXds59XEGca2XWedW1JPJ7fqFnuGK0dkuBc6kcqjbvJ2bT3+aCrMIJBEEGCOS6FyecVhWgqpSBTYFhCgQrAUoSsCDXKwOCRpccdwx8VAEj669+iB0WFoUqVnLnBozcQBO8mBiqAVIPKKEdFR2TUaHNFctvCH4ENIGjzeHV5jsWPQtb2/KwjPrU2P8XAkdhUn2wmldcXk3xiSS27d+HExM4ofpFEU+ypNdF1G0gCDRY/HN3SA8uo9uCVK1NGdJjsfmNQRwFx4w54qm+leCqiS6lagJmkx0nU1BHAXXjDnJWnZrI4WZ9c9G1tR/R02ll9zokuNNxxYG78ZyWfs2yGtVZRZ8T3Bo4TmeQEnsW79pbS11boqf8AKoDoaY/T8buZdOOsBZyfRcI2zJk6kk7zj4pJJKTpSoeEyUovZbKRqsFd0Upl+BMgYhsDGCcCdyAYW55s9mmYq2luAwllnxE8DUJMcAsBau3RWqVH2h7ZaTg5hD6bWjBrL7ZDYECDB4LKVROWTbY6UJJJkhFC2PaLoMt/CcR2ajHHCFrWJ4NB9VohzajWuAOEPY664bsQ4RjosIFbP2a638RR/wBSi5zeL6RFRo7g7vSlwaacmmiyjbMcQJ36oqwCneFy83kT+6yAF0P2dsWN85Dz9+iwnKkd+lcpFu3zejGXHCTjMYEnzWGbKZmGEDISQZGUzhC17ZSfUcIwLjdbwBOJ5QCeQCyrfUaHltMktbAmfiIzdwnuV6ftjQarTnZTZrNUD74N12skPBnMERiDuK3rHZG3er1RJkaTrHBZVAzBaTyPiCjn28twiN6c02PTaQM1snqjvVrn9B94IvOBbHH8XZ68ErKyTJwAxJ3BZtvtXSPnIZN4Ae5SSt0LTWLHpEGXHtnVEWI3nFx9ncs6pVwAA/wtKx04AJ5x9fei1rASl0dBZqePvmffBc7tmqSy3P8AzWaiP6TfeO9b1leeWBJ985XJbRqf+wD/APWtVSpzAaW+ZURWTD+Q8JHPgkYjPRae0m9Ixtobr1ag3PGvI/RZgR+yrUGOLX403i68eTuxbPycaM8hMjtoWM0nlpxGbTvGhQTmqk7Akx4Gknw8M0/STn4KpOih2XgpKkKQcUqEWQl0YUA5Ta8IAIaHdC5sdW+1xduN1wA7ZPch7hRFO1RTfTibxYZnK7OnGVSEgIXCldKsSRYB/wBnNqmy1xWFMPgEQTBAcIJadHRrjmVrM2WyuJslQvOJNGpDa41N35ao4iCuaCdriDIwIxBGBB3gqJRvJUZNB72EEgggjAgggg7iDiCorRofaEVAGWyn0wAhtVvVrsHB/wA44O71A2LpKjm2W9XaACDduOjUFpzIykZqKa5N46iYEEk5bEggggwQcCDuITILLKFZzHXmOLXb2ktPKRjCvdaWP/m0muP4mfdP5y0XXf1MJ4oRKUCcU+Qh1gY7+VVH6asUncg+TTPa5p4IS1WSpTMVGOYTleEA8WnJw4hTVrLU9jHNa8hpBlubTOcsPVnjEp2zOWkugRjScgTGcCUd9nbWKdqov0FRod+l3Ud4OK660MdfFJjntZToUHNYxzqbZeDecQwi8SRrKyNoPIJa834/H95poakkdkIT3YEtF1usorbPLK76MfC9zRyBgeELpnAU6YpjLAuPkO0+8Fi0NpGpaDWLQSQJuggXoDRmTiQFoVSX1m0xobzzpe3cYmO0rCUXeT0NKlH7B9rVbrfzEQODfmPviuccNy1tuSKro+VxA3YYIezU23iTg1uepP5Rw8+1VEy1PdITT0TbxHWdBjcN548OSiKwOMnHdkeKFtFcudePdw3Ihrgzq578cju98Vd0KLv6CdqVwxvRNzMF/oPVZLRmewep98VY4lziTiXT9VVa6mg5D1KuKo1k0QY2Xe8vfmtukZgcvDE+qx7K3DjkPfNbNlb4D35K5GMcsNtVpuWeodbro7iAO9c39ohdstiZ+R7z/W4ELT+0lW7ZyNXFo/5f8UB9verVpUtKdCm3z/ZRHlGP8h5o5unmBv3mB2k5Diui2xZLFTpNbSqvq1/mLSOiG/SSN0Gd65tSa5auNuzBOkbez6grM6B5hw/lO3flPvyWZWolji1wgjAqtp1HYt1sWpkYCuwf3j33FQ/b9C5MIhRuqx7CCQRBGBB0KYK7EVkJBWpi1FgVp5UujSuJgGWSo3oa7SQCeiujUw6TG/BBBFWSxF7ajpi429ETP0QhCS7GPKeUwSQIleTXimSQA8orZ20alF16m4tdiJG46Y5hCymQ1Yzd2VbrO9z/AOK6QOeZFVhHUMkklkQ4GfDAI637CqMZ0rC2vR/1aWIH62ZsPgFysI7ZW1K1nffovc06xkRuc04HtWcodoqM2i4bxkkVuU7fZLX/ADm/wtc//bTE0nnfUp/Lz8ULtbYlagLz2g0z8NWmb1Mg5Yj4e3xUfZuppmak89UpKNU9UoKfDO9qOi1c7JZ/+YStEOPaPKFTtExaaZ32Sj4X/qnYfh59nvNQa6PxBbXZ7rhGRHs+KF2JUqstFRxM0hiMMSSMBO4H0R1tMgfpMc4H08EC4OL2ODg1jSXu5XdeGKro1XkEtO06fTPlxY4OeTPwmCdcQBMd6nUIIAEPbmS0gSd4jCOaA+0NlEl3448MT4wVz1npljrzXOaBi6DE7huk/votIwUlaMJycZbWjqxSYOsHFpPwh413yPfchTQdpB5EfVZVP7Q1Qeu1rx3GN0opm2bMRi1zTu63/Ewl6UkTui+GFZcz5INzpJ7ldaamE6nAKmizGO1XFGmo+guyMx9+/wDC2LOMBxPeP8BZllb9B5LYpNjkB79VMmKKANqTUrWan+KqJ5BzR6uWZ9uat621eBa3ua0ecrY2Y2/tGg38DQTzuuefF4XMbfrX7TWdvqP7rxARD5fo5NV3JmeSkEk62Mhw6FdQtBaQ5pgjIqhPCTQHR1abbWy+2BWaOsPxe96wnNIJBwIzBTWa0OpuDmGCPcHeFvVKbLWy+yG1QOsN/vQqPj9D5MKU8pPEEgiCMwUkxDpJJIAN2faWsbWDp69Mtbz0QSIsFkNVxAIENLsdw0Q6XYCSSlPKYDwmhJOgB4STEp0gGSCSSYClauxftBXsx+7dLD8VN3WYd/V05iFlJJNJhwdhTp2K2fyyLJXPyOxovO5p+Xw5FY+2dl1rPLa1MtnJwxY79LsuzNY5WpT2/XFB9nc+9ScAId1i2DIunMZZZcFGxrgtTdUdbtj/AORQ42Wn4Pb9Va5mHI+f+VDbQ++svGy+RYUQRg7v99yy6R26PxM+q4RGodx1O/k5ZttYTTc3mPUefgtWtTxcOR78PRCvzneA7u/YqvyapXaM6n97Z2/ib1f7f2hYNsZoMh4neutoMEFoEa9uvqsDaVCCQr05ZZGvB7UzCexQLUXUYn/gahyY7+0ro3HFtfRoVDLuAw9+HirqVMxO+I7cAkks+jqeWaVkowQN2PotCliXDTzwy70kli2aRB/sj1rbaKujGvHiGjwYuHrPlxO/HvxTpLSHyZ50v9K06SS1JEnlJJADK2y2h1Nwe0wR7g7wkkgDoX0qdsZebDarRiN/PeOK5+rScxxa4EEZhJJZxxLaU+LIhye+kkrJNf7MOHSuG+m/0WU1ySSmssfRKQlKdJOhDSpBJJIBinBSSQAwcmLwnSToBr4S6RJJFANfKg44JJIA9H246Klh40CPBiPuabx9UklyvhHoaHD+wOs3InUfQ/VAvbB5Eg8jl4EJ0k1wbLkai3H3yQW1KIAm7POeYyhMkpi/ca6i/wCZg1qrhlhyAHiMUMYOJz44pJLrR5sj/9k=",caption="My Favorite Map Driftwoods(Night)") 
   
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

elif page == 'Exploratory Data Analysis':
    st.title("🕹️ Exploratory Data Analysis (EDA)")
    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])
#Histograms
    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:",['color','team_name','team_region','core_shots','core_goals','core_saves','core_assists','core_score','core_shooting_percentage','winner'])
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col}"
        if st.checkbox("Show by Color"):
            st.plotly_chart(px.histogram(df, x=h_selected_col, color=("color"), title=chart_title, barmode='overlay'))
        else:
            st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))
#Boxplots
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the Box Plot:",['color','team_name','team_region','core_shots','core_goals','core_saves','core_assists','core_score','core_shooting_percentage','winner'])
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col}"
        if st.checkbox("Show by Color"):
            st.plotly_chart(px.box(df, x=h_selected_col, color="color", title=chart_title, boxmode='overlay'))
        else:
            st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))
#Scatterplots
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", ['color','team_name','team_region','core_shots','core_goals','core_saves','core_assists','core_score','core_shooting_percentage','winner'])
        selected_col_y = st.selectbox("Select y-axis variable:", ['color','team_name','team_region','core_shots','core_goals','core_saves','core_assists','core_score','core_shooting_percentage','winner'])
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='color', title=chart_title))
#Countplots
    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical variable:", ['color','team_name','team_region','core_shots','core_goals','core_saves','core_assists','core_score','core_shooting_percentage','winner'])
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.histogram(df, x=selected_col, color='color', title=chart_title))


# Model Training and Evaluation Page
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")
    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X=df.drop(columns= ['color','team_name','team_region','winner']).values
    y=df['winner']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

elif page == "About":
    st.title("🎧 About")
    st.write("""This is an example of a multi-page Streamlit app using a Rocket League dataset.
    The app showcases knowledge on how each page should be edited to help show case the dataset""")
    url = "https://www.kaggle.com/datasets/dylanmonfret/rlcs-202122"
    st.write("For further info on this dataset check out this [link](%s)" % url)