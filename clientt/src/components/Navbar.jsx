import React from 'react';
import "../styles.css";
import { Link } from "react-router-dom";
import loopNav from "../img/loopNav.png";
import statsImg from "../img/graphNav.png";
import messageNav from "../img/messageNav.png";
import wallet from "../img/Wallet.png";




function Navbar() {
  return (
    <div className="nav">
        <span>AVAVAVAVA</span>
        <div className="navA">
            <div className="navTop">
                <Link to="/search" className="nav-item">
                    <img src={loopNav} alt="loopNav"/>
                    <p>Рекомендации</p>
                </Link>
                <hr/>
                <Link to="#" className="nav-item">
                    <img src={statsImg} alt="graphNav"/>
                    <p>Статистика</p>
                </Link>
                <hr/>
                <Link to="#" className="nav-item">
                    <img src={messageNav} alt="messageNav"/>
                    <p>Чат-Бот</p>
                </Link>
                <hr/>
            </div>
            <div className="navBottom">
                <a to="#" className="nav-item">
                    <p>Личный кабинет</p>
                </a>
                <hr/>
                <a to="#" className="nav-item">
                    <img src={wallet} alt="messageNav"/>
                    <p>Тарифы</p>
                </a>
            </div>
        </div>
    </div> 
  )
}

export default Navbar
