import React from "react";
import "../App.css";

const Calendar = () => {
  const today = new Date();
  const year = today.getFullYear();
  const month = today.getMonth();

  const firstDayOfMonth = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();

  const weekDays = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"];
  const adjustedStart = (firstDayOfMonth + 6) % 7;


  const days = [];

  for (let i = 0; i < adjustedStart; i++) {
    days.push(<div key={`empty-${i}`} className="day empty" />);
  }

  for (let i = 1; i <= daysInMonth; i++) {
    days.push(
      <div key={i} className="day">
        <span>{i}</span>
      </div>
    );
  }

  return (
    <div className="calendar-container">
      <div className="weekdays">
        {weekDays.map((d, i) => (
          <div key={i} className="weekday">
            {d}
          </div>
        ))}
      </div>

      <div className="days-grid">{days}</div>
    </div>
  );
};

export default Calendar;
