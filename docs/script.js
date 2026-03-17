const toggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

if (toggle && navLinks) {
  toggle.addEventListener('click', () => {
    const isOpen = navLinks.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(isOpen));
  });

  navLinks.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    });
  });
}

const heroImage = document.getElementById("hero-slideshow-image");

if (heroImage) {
  const heroImages = [
    "images/hero_01.png",
    "images/hero_02.png",
    "images/hero_03.png",
    "images/hero_04.png",
    "images/hero_05.png"
  ];

  let heroIndex = 0;

  setInterval(() => {
    heroImage.classList.add("is-fading");

    setTimeout(() => {
      heroIndex = (heroIndex + 1) % heroImages.length;
      heroImage.src = heroImages[heroIndex];
      heroImage.classList.remove("is-fading");
    }, 800);
  }, 3000);
}
