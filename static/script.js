// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// DOM Loaded Event
document.addEventListener('DOMContentLoaded', () => {
    const hamburger = document.getElementById('hamburger');
    const menu = document.getElementById('menu');

    // Toggle Menu
    hamburger.addEventListener('click', () => {
        menu.classList.toggle('hidden'); // Show or hide the menu
    });
});

// Tombol hamburger
document.getElementById("hamburger").addEventListener("click", function () {
    const menu = document.getElementById("menu");
    menu.classList.toggle("hidden");
    menu.classList.toggle("active");
});

// Menutup menu ketika tautan ditekan
const menu = document.getElementById("menu");
const menuItems = menu.querySelectorAll("a");

menuItems.forEach(item => {
    item.addEventListener("click", () => {
        menu.classList.add("hidden");
        menu.classList.remove("active");
    });
});  

