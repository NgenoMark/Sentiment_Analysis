const form = document.getElementById("search-form");
const input = document.getElementById("movie-title");
const resultsDiv = document.getElementById("results");
const historyDiv = document.getElementById("history");
let searchHistory = [];

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const title = input.value.trim();
  if (!title) return;

  updateSearchHistory(title);
  await fetchRecommendations(title);
});

async function fetchRecommendations(title) {
  try {
    const response = await fetch(`http://127.0.0.1:5000/recommend?title=${encodeURIComponent(title)}`);
    const data = await response.json();

    resultsDiv.innerHTML = "";

    if (data.length === 0) {
      resultsDiv.innerHTML = `<p>No recommendations found for "${title}".</p>`;
      return;
    }

    data.forEach(movie => {
      const movieDiv = document.createElement("div");
      movieDiv.classList.add("movie");

      const genreList = movie.genres.join(", ");

      movieDiv.innerHTML = `
        <img src="${movie.poster_url || fallbackImage}" onerror="this.src='${fallbackImage}'" alt="Poster">
        <div class="movie-info">
          <h3>${movie.title}</h3>
          <p><strong>Genres:</strong> ${genreList}</p>
          <p>${movie.overview}</p>
        </div>
      `;

      resultsDiv.appendChild(movieDiv);
    });
  } catch (error) {
    resultsDiv.innerHTML = `<p>Error fetching recommendations. Please try again later.</p>`;
    console.error("Error:", error);
  }
}

function updateSearchHistory(title) {
  if (!searchHistory.includes(title)) {
    searchHistory.unshift(title);
    if (searchHistory.length > 10) searchHistory.pop();
    renderSearchHistory();
  }
}

function renderSearchHistory() {
  historyDiv.innerHTML = "<h3>Search History</h3>";
  const list = document.createElement("ul");
  searchHistory.forEach(item => {
    const li = document.createElement("li");
    li.textContent = item;
    li.addEventListener("click", () => {
      input.value = item;
      fetchRecommendations(item);
    });
    list.appendChild(li);
  });
  historyDiv.appendChild(list);
}