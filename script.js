document.addEventListener('DOMContentLoaded', () => {
  const filter = document.getElementById('scenario-filter');
  if (!filter) return;
  const apply = () => {
    const value = filter.value;
    document.querySelectorAll('.scenario-result').forEach(row => {
      row.hidden = value !== 'all' && row.dataset.scenario !== value;
    });
    document.querySelectorAll('#iteration-table tbody tr').forEach(row => {
      row.hidden = value !== 'all' && row.dataset.scenario !== value;
    });
  };
  filter.addEventListener('change', apply);
  const tocLinks = [...document.querySelectorAll('.toc a')];
  const sections = tocLinks.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          tocLinks.forEach(link => link.classList.toggle('active', link.getAttribute('href') === '#' + entry.target.id));
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    sections.forEach(section => observer.observe(section));
  }
});
