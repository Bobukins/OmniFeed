const daysContainer = document.getElementById("days");
const monthYear = document.getElementById("monthYear");
const months = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"];
const currentYear = 2025;
const currentMonth = 2;

function generateCalendar(year, month) {
    daysContainer.innerHTML = "";
    monthYear.textContent = `${months[month]} ${year}`;
    const firstDay = new Date(year, month, 1).getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    
    for (let i = 0; i < (firstDay === 0 ? 6 : firstDay - 1); i++) {
        daysContainer.appendChild(document.createElement("div"));
    }
    
    for (let i = 1; i <= daysInMonth; i++) {
        const day = document.createElement("div");
        day.textContent = i;
        day.classList.add("day");
        daysContainer.appendChild(day);
    }
}

generateCalendar(currentYear, currentMonth);



