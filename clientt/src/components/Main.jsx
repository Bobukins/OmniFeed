import React, { useState } from "react";
import TopNav from "./TopNav";
import fire from "../img/🔥.png";
import Calendar from "./Calendar";

const Main = () => {
  const [query, setQuery] = useState("");
  const handleSearch = () => {
    console.log("searching for", query);
  };

  return (
    <div class="main">
      <TopNav query={query} setQuery={setQuery} onSearch={handleSearch} />{" "}
      <hr class="hrMain" />
      <div class="areYouAuth">
        <a class="areYouAuthBtn" href="./index.html">
          Настроить виджеты
        </a>
        <div class="areYouAuth_addAccount">
          <div>
            <img
              src="https://storage.yandexcloud.net/olimp-bucket/photo_2025-03-30_00-02-08%20(2).jpg"
              alt="stars"
            />
            <div class="areYouAuth_addAccountTxt">
              <h5>Добавьте аккаунт, чтобы увидеть статистику</h5>
              <label>
                С помощью нее вы сможете увидеть свои охваты и потенциальных
                конкурентов
              </label>
            </div>
          </div>
          <button>Добавить аккаунт</button>
        </div>
      </div>
      <div class="allallbody">
        <div class="allBody">
          <div class="allStats">
            <div class="allStatsA">
              <h2>Сводка статистики</h2>
              <div class="statsA">
                <form class="numViews" method="POST">
                  <div class="numViewsTxt">
                    <h4>КОЛИЧЕСТВО ПРОСМОТРОВ</h4>
                    <div class="numViewsCount">
                      <label name="numberOfViews1">174 543.09</label>
                      <p>
                        +154
                        <img src={fire} />
                      </p>
                    </div>
                  </div>
                </form>
                <form class="numViews" method="POST">
                  <div class="numViewsTxt">
                    <h4>КОЛИЧЕСТВО ПРОСМОТРОВ</h4>
                    <div class="numViewsCount">
                      <label name="numberOfViews2">174 543.09</label>
                      <p>
                        +154
                        <img src={fire} />
                      </p>
                    </div>
                  </div>
                </form>
                <form class="numViews" method="POST">
                  <div class="numViewsTxt">
                    <h4>КОЛИЧЕСТВО ПРОСМОТРОВ</h4>
                    <div class="numViewsCount">
                      <label name="numberOfViews3">174 543.09</label>
                      <p>
                        +154
                        <img src={fire} />
                      </p>
                    </div>
                  </div>
                </form>
                <form class="numViews" method="POST">
                  <div class="numViewsTxt">
                    <h4>КОЛИЧЕСТВО ПРОСМОТРОВ</h4>
                    <div class="numViewsCount">
                      <label name="numberOfViews4">174 543.09</label>
                      <p>
                        +154
                        <img src={fire} />
                      </p>
                    </div>
                  </div>
                </form>
              </div>
            </div>

            <div>
              <h2>Подходящие темы для публикаций</h2>
              <div class="statsA">
                <form class="numViews" method="POST">
                  <div class="secStatistic">
                    <div class="secStatisticViews1">
                      <h5>Тема</h5>
                      <p name="theme5">Цветочный маркет</p>
                      <p name="theme6">Праздник</p>
                      <p name="theme7">Поздравления</p>
                    </div>
                    <div class="secStatisticViews">
                      <h5>Просмотры</h5>
                      <label name="numberOfViews1">+35%</label>
                      <label name="numberOfViews2">+35%</label>
                      <label name="numberOfViews3">+35%</label>
                    </div>
                  </div>
                </form>
                <form class="numViews" method="POST">
                  <div class="secStatistic">
                    <div class="secStatisticViews1">
                      <h5>Тема</h5>
                      <p name="theme8">Цветочный маркет</p>
                      <p name="theme9">Праздник</p>
                      <p name="theme10">Поздравления</p>
                    </div>
                    <div class="secStatisticViews">
                      <h5>Просмотры</h5>
                      <label name="numberOfViews1">+35%</label>
                      <label name="numberOfViews2">+35%</label>
                      <label name="numberOfViews3">+35%</label>
                    </div>
                  </div>
                </form>
              </div>
            </div>
          </div>

          <form class="lastSeen" method="POST">
            <h5>Недавно просмотренные</h5>
            <ul class="lastSeenFirtsBlock">
              <li class="lastseenCard">
                <div>
                  <p>
                    <span style={{ color: "#000" }} name="nameAccLast1">
                      blumcrypto
                    </span>
                    <br />
                    <span
                      style={{ color: "rgb(114, 114, 114);" }}
                      name="nameAccLastSubs1">
                      29.1M
                    </span>{" "}
                    подписчиков
                  </p>
                </div>
                <a href="#">Добавить</a>
              </li>
              <li class="lastseenCard">
                <div>
                  <p>
                    <span style={{ color: "#000" }} name="nameAccLast1">
                      blumcrypto
                    </span>
                    <br />
                    <span
                      style={{ color: "rgb(114, 114, 114)" }}
                      name="nameAccLastSubs1">
                      29.1M
                    </span>{" "}
                    подписчиков
                  </p>
                </div>
                <a href="#">Добавить</a>
              </li>
              <li class="lastseenCard">
                <div>
                  <p>
                    <span style={{ color: "#000" }} name="nameAccLast1">
                      blumcrypto
                    </span>
                    <br />
                    <span
                      style={{ color: "rgb(114, 114, 114)" }}
                      name="nameAccLastSubs1">
                      29.1M
                    </span>{" "}
                    подписчиков
                  </p>
                </div>
                <a href="#">Добавить</a>
              </li>
              <li class="lastseenCard">
                <div>
                  <p>
                    <span style={{ color: "#000" }} name="nameAccLast1">
                      blumcrypto
                    </span>
                    <br />
                    <span
                      style={{ color: "rgb(114, 114, 114);" }}
                      name="nameAccLastSubs1">
                      29.1M
                    </span>{" "}
                    подписчиков
                  </p>
                </div>
                <a href="#">Добавить</a>
              </li>
              <li class="lastseenCard">
                <div>
                  <p>
                    <span style={{ color: "#000" }} name="nameAccLast1">
                      blumcrypto
                    </span>
                    <br />
                    <span
                      style={{ color: "rgb(114, 114, 114);" }}
                      name="nameAccLastSubs1">
                      29.1M
                    </span>{" "}
                    подписчиков
                  </p>
                </div>
                <a href="#">Добавить</a>
              </li>
              <li class="lastseenCard">
                <div>
                  <p>
                    <span style={{ color: "#000" }} name="nameAccLast1">
                      blumcrypto
                    </span>
                    <br />
                    <span
                      style={{ color: "rgb(114, 114, 114);" }}
                      name="nameAccLastSubs1">
                      29.1M
                    </span>{" "}
                    подписчиков
                  </p>
                </div>
                <a href="#">Добавить</a>
              </li>
              <li class="lastseenCard">
                <div>
                  <p>
                    <span style={{ color: "#000" }} name="nameAccLast1">
                      blumcrypto
                    </span>
                    <br />
                    <span
                      style={{ color: "rgb(114, 114, 114)" }}
                      name="nameAccLastSubs1">
                      29.1M
                    </span>{" "}
                    подписчиков
                  </p>
                </div>
                <a href="#">Добавить</a>
              </li>
            </ul>
          </form>
        </div>

        <div class="mainBodyRight" name="mainBody">
          <form class="calendarBlock" method="POST">
            <div class="navCal">
              <h2 id="monthYear">Март 2025</h2>
            </div>
            <Calendar />
          </form>
          <h1>Полезные статьи</h1>
          <form class="lastStats" method="POST">
            <ul class="stats">
              <li class="stat">
                <div>
                  <h5 name="statHeader">
                    Час пробил: в какое время публиковать посты в разных
                    соцсетях, чтоб «залетели»
                  </h5>
                  <p name="stattxt">
                    Как запрещённые, так и разрешённые в России соцсети кричат
                    начинающему блогеру....
                  </p>
                </div>
              </li>
              <li class="stat">
                <div>
                  <h5 name="statHeader">
                    Час пробил: в какое время публиковать посты в разных
                    соцсетях, чтоб «залетели»
                  </h5>
                  <p name="stattxt">
                    Как запрещённые, так и разрешённые в России соцсети кричат
                    начинающему блогеру....
                  </p>
                </div>
              </li>
              <li class="stat">
                <div>
                  <h5 name="statHeader">
                    Час пробил: в какое время публиковать посты в разных
                    соцсетях, чтоб «залетели»
                  </h5>
                  <p name="stattxt">
                    Как запрещённые, так и разрешённые в России соцсети кричат
                    начинающему блогеру....
                  </p>
                </div>
              </li>
            </ul>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Main;
