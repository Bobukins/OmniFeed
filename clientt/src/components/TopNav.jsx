import React from "react";
import searchpng from "../img/Search.png";
import "../styles.css";
import { useNavigate, useLocation } from "react-router-dom";

const TopNav = ({ query, setQuery, onSearch }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleSearchClick = () => {
    if (location.pathname !== "/search") {
      navigate(`/search?q=${encodeURIComponent(query)}`);
    } else {
      onSearch();
    }
  };

  const handleEnterKey = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (location.pathname !== "/search") {
        navigate("/search");
      }
      onSearch();
    }
  };

  return (
    <form
      className="main_nav"
      method="POST"
      onSubmit={(e) => e.preventDefault()}>
      <div className="search" name="search">
        <input
          placeholder="Кто вас интерисует?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleEnterKey}
        />
        <img
          src={searchpng}
          alt="loop"
          onClick={handleSearchClick}
          style={{ cursor: "pointer" }}
        />
      </div>

      <div className="notificationAccount">
        <div className="notificationAccountBell">
          <img
            src="https://storage.yandexcloud.net/olimp-bucket/photo_2025-03-30_00-02-07.jpg"
            alt="bell"
          />
          <p id="notificationCount"></p>
        </div>
        <hr className="notificationAccountHR" />
        <img
          src="https://storage.yandexcloud.net/olimp-bucket/photo_2025-03-30_00-02-06.jpg"
          alt="accountIcon"
        />
      </div>
    </form>
  );
};

export default TopNav;
