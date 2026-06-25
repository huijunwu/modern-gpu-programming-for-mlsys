// Language switcher for multi-language site
document.addEventListener('DOMContentLoaded', function() {
  const currentPath = window.location.pathname;
  const isChinese = currentPath.startsWith('/zh/');
  
  // Create language switcher dropdown
  const switcher = document.createElement('div');
  switcher.className = 'language-switcher';
  switcher.innerHTML = `
    <label for="language-select">Language / 语言:</label>
    <select id="language-select">
      <option value="en" ${!isChinese ? 'selected' : ''}>English</option>
      <option value="zh" ${isChinese ? 'selected' : ''}>简体中文</option>
    </select>
  `;
  
  // Insert at the top of the page
  const container = document.querySelector('body > div') || document.body;
  container.insertBefore(switcher, container.firstChild);
  
  // Handle language change
  document.getElementById('language-select').addEventListener('change', function() {
    const selectedLang = this.value;
    const currentPath = window.location.pathname;
    
    if (selectedLang === 'en' && isChinese) {
      // Switch from Chinese to English - remove /zh/ prefix
      window.location.href = currentPath.replace('/zh/', '/');
    } else if (selectedLang === 'zh' && !isChinese) {
      // Switch from English to Chinese - add /zh/ prefix
      window.location.href = '/zh/' + currentPath.substring(1);
    }
  });
});
