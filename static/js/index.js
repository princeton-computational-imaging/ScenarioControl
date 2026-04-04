window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Navbar burger toggle
    $(".navbar-burger").click(function() {
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");
    });
});

// ScenarioControl interactive galleries
(function () {
    var conditionImages = [
        "img_cond_1.jpg",
        "img_cond_2.jpg",
        "img_cond_3.jpg",
        "img_cond_4.jpg",
        "img_cond_5.jpg",
        "img_cond_6.jpg",
        "img_cond_7.jpg",
        "img_cond_8.jpg"
    ];

    var sceneCountPerCondition = 4;
    var selectedCondition = 1;
    var selectedScene = 1;

    var thumbsContainer = document.getElementById("img-cond-thumbs");
    var condImageEl = document.getElementById("selected-cond-image");
    var sceneImageEl = document.getElementById("selected-scene-image");
    var sceneIndexLabel = document.getElementById("scene-index-label");
    var prevBtn = document.getElementById("scene-prev");
    var nextBtn = document.getElementById("scene-next");

    var projConditionIndices = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14];
    var selectedProjCondition = 1;
    var projShowingGenerated = false;

    var projThumbsContainer = document.getElementById("img-cond-proj-thumbs");
    var projResultImageEl = document.getElementById("selected-proj-result-image");
    var projPhaseLabel = document.getElementById("proj-phase-label");

    var textCondConfigs = [
        { intersection: "four-way", curvature: "straight", pedestrians: "no",  lanes: "more", sources: [1, 7], prompt: "The scene is a four-way intersection featuring multiple lanes in each direction. Multiple vehicles exist." },
        { intersection: "T",        curvature: "straight", pedestrians: "no",  lanes: "more", sources: [2, 8], prompt: "The scene is a T-intersection. Multiple vehicles exist." },
        { intersection: "none",     curvature: "curved",   pedestrians: "yes", lanes: "more", sources: [3, 9], prompt: "The scene depicts a multi-lane road with a curved layout. There are also pedestrians near the road." },
        { intersection: "none",     curvature: "straight", pedestrians: "no",  lanes: "more", sources: [4],    prompt: "The scene depicts a multi-lane road with a left-turning ahead." },
        { intersection: "none",     curvature: "straight", pedestrians: "no",  lanes: "less", sources: [5],    prompt: "The scene depicts a narrow, vertically oriented road." },
        { intersection: "complex",  curvature: "straight", pedestrians: "no",  lanes: "more", sources: [6],    prompt: "The scene depicts a multi-lane intersection with a complex road topology." }
    ];
    var textCondCategories = ["intersection", "curvature", "pedestrians", "lanes"];
    var textCondSelection = { intersection: "four-way", curvature: "straight", pedestrians: "no", lanes: "more" };
    var textCondCurrentConfig = null;
    var textCondPage = 0;

    var textCondPromptEl = document.getElementById("selected-text-cond-prompt");
    var textCondCarouselTrack = document.getElementById("text-cond-carousel-track");
    var textCondCarousel = document.getElementById("text-cond-carousel");
    var textCondPageLabel = document.getElementById("text-cond-page-label");
    var textCondPrevBtn = document.getElementById("text-cond-prev");
    var textCondNextBtn = document.getElementById("text-cond-next");

    function getSceneImagePath(condIndex, sceneIndex) {
        return "media/img_cond/scene_img_cond_" + condIndex + "_" + sceneIndex + ".jpg";
    }

    function getProjOrigPath(condIndex) {
        return "media/img_cond_proj/img_cond_proj_" + condIndex + "_orig.jpg";
    }

    function getProjMatchedPath(condIndex) {
        return "media/img_cond_proj/img_cond_proj_" + condIndex + "_orig_matched.jpg";
    }

    function getProjGeneratedPath(condIndex) {
        return "media/img_cond_proj/img_cond_proj_" + condIndex + ".jpg";
    }

    function renderThumbs() {
        thumbsContainer.innerHTML = "";
        conditionImages.forEach(function (filename, idx) {
            var condIndex = idx + 1;
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "thumb-btn" + (condIndex === selectedCondition ? " active" : "");
            btn.setAttribute("aria-label", "Select conditioning image " + condIndex);

            var img = document.createElement("img");
            img.src = "media/img_cond/" + filename;
            img.alt = "Conditioning preview " + condIndex;

            btn.appendChild(img);
            btn.addEventListener("click", function () {
                selectedCondition = condIndex;
                selectedScene = 1;
                updateView();
            });

            btn.addEventListener("mouseenter", function () {
                btn.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            });

            btn.addEventListener("focus", function () {
                btn.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            });

            thumbsContainer.appendChild(btn);
        });
    }

    function renderProjThumbs() {
        if (!projThumbsContainer) {
            return;
        }

        projThumbsContainer.innerHTML = "";
        projConditionIndices.forEach(function (condIndex) {
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "thumb-btn" + (condIndex === selectedProjCondition ? " active" : "");
            btn.setAttribute("aria-label", "Select projection conditioning image " + condIndex);

            var img = document.createElement("img");
            img.src = getProjOrigPath(condIndex);
            img.alt = "Projection conditioning preview " + condIndex;

            btn.appendChild(img);
            btn.addEventListener("click", function () {
                selectedProjCondition = condIndex;
                projShowingGenerated = false;
                updateProjView();
            });

            btn.addEventListener("mouseenter", function () {
                btn.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            });

            btn.addEventListener("focus", function () {
                btn.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
            });

            projThumbsContainer.appendChild(btn);
        });
    }

    function updateView() {
        condImageEl.src = "media/img_cond/" + conditionImages[selectedCondition - 1];
        sceneImageEl.src = getSceneImagePath(selectedCondition, selectedScene);
        sceneIndexLabel.textContent = "Sample " + selectedScene + " / " + sceneCountPerCondition;
        renderThumbs();
    }

    if (prevBtn) {
        prevBtn.addEventListener("click", function () {
            selectedScene = selectedScene === 1 ? sceneCountPerCondition : selectedScene - 1;
            updateView();
        });
    }
    if (nextBtn) {
        nextBtn.addEventListener("click", function () {
            selectedScene = selectedScene === sceneCountPerCondition ? 1 : selectedScene + 1;
            updateView();
        });
    }

    function updateProjView() {
        if (!projResultImageEl) {
            return;
        }

        projResultImageEl.src = projShowingGenerated
            ? getProjGeneratedPath(selectedProjCondition)
            : getProjMatchedPath(selectedProjCondition);

        if (projPhaseLabel) {
            projPhaseLabel.textContent = projShowingGenerated ? "Generated Scenario" : "Input \u2192 Projection";
        }

        renderProjThumbs();
    }

    function getTextCondSamples(config) {
        var samples = [];
        config.sources.forEach(function (srcId) {
            for (var s = 1; s <= 3; s++) {
                samples.push("media/text_cond/text_cond_" + srcId + "_" + s + ".jpg");
            }
        });
        return samples;
    }

    function findExactConfig(selection) {
        for (var i = 0; i < textCondConfigs.length; i++) {
            var c = textCondConfigs[i];
            var match = true;
            for (var j = 0; j < textCondCategories.length; j++) {
                if (c[textCondCategories[j]] !== selection[textCondCategories[j]]) {
                    match = false;
                    break;
                }
            }
            if (match) return c;
        }
        return null;
    }

    function findNearestConfig(selection, lockedCat) {
        var bestConfig = textCondConfigs[0];
        var bestDist = Infinity;
        for (var i = 0; i < textCondConfigs.length; i++) {
            if (lockedCat && textCondConfigs[i][lockedCat] !== selection[lockedCat]) continue;
            var dist = 0;
            for (var j = 0; j < textCondCategories.length; j++) {
                if (textCondConfigs[i][textCondCategories[j]] !== selection[textCondCategories[j]]) dist++;
            }
            if (dist < bestDist) {
                bestDist = dist;
                bestConfig = textCondConfigs[i];
            }
        }
        return bestConfig;
    }

    function computeAvailability(selection) {
        var availability = {};
        textCondCategories.forEach(function (targetCat) {
            availability[targetCat] = {};
            var seen = {};
            textCondConfigs.forEach(function (c) {
                var val = c[targetCat];
                if (seen[val]) return;
                if (selection[targetCat] === val) {
                    availability[targetCat][val] = true;
                    seen[val] = true;
                    return;
                }
                // Find nearest config that has this value (locked)
                var tempSel = { intersection: selection.intersection, curvature: selection.curvature,
                    pedestrians: selection.pedestrians, lanes: selection.lanes };
                tempSel[targetCat] = val;
                var best = findNearestConfig(tempSel, targetCat);
                // Count how many OTHER categories would change
                var otherChanges = 0;
                textCondCategories.forEach(function (cat) {
                    if (cat !== targetCat && best[cat] !== selection[cat]) otherChanges++;
                });
                availability[targetCat][val] = otherChanges < 2;
                seen[val] = true;
            });
        });
        return availability;
    }

    function updateTextCondView() {
        if (!textCondCarouselTrack || !textCondPromptEl) return;
        var samples = getTextCondSamples(textCondCurrentConfig);
        var totalPages = Math.ceil(samples.length / 3);

        // Update carousel images
        var imgs = textCondCarouselTrack.querySelectorAll("img");
        for (var i = 0; i < 3; i++) {
            var idx = textCondPage * 3 + i;
            if (idx < samples.length) {
                imgs[i].src = samples[idx];
                imgs[i].style.display = "";
            } else {
                imgs[i].style.display = "none";
            }
        }

        // Update prompt
        textCondPromptEl.textContent = textCondCurrentConfig.prompt;

        // Update page label and chevron visibility
        if (totalPages > 1) {
            textCondCarousel.classList.add("has-pages");
            textCondPageLabel.textContent = "Page " + (textCondPage + 1) + " / " + totalPages;
            textCondPageLabel.style.display = "";
        } else {
            textCondCarousel.classList.remove("has-pages");
            textCondPageLabel.style.display = "none";
        }

        // Update pill states
        var availability = computeAvailability(textCondSelection);
        var pills = document.querySelectorAll("#text-cond-configurator .config-pill");
        for (var i = 0; i < pills.length; i++) {
            var pill = pills[i];
            var cat = pill.closest(".config-category").getAttribute("data-category");
            var val = pill.getAttribute("data-value");
            if (textCondSelection[cat] === val) {
                pill.classList.add("active");
            } else {
                pill.classList.remove("active");
            }
            if (availability[cat] && availability[cat][val]) {
                pill.classList.remove("unavailable");
            } else {
                pill.classList.add("unavailable");
            }
        }
    }

    function resolveTextCondConfig(lockedCat) {
        var config = findExactConfig(textCondSelection);
        if (!config) {
            config = findNearestConfig(textCondSelection, lockedCat);
            textCondCategories.forEach(function (cat) {
                textCondSelection[cat] = config[cat];
            });
        }
        textCondCurrentConfig = config;
        textCondPage = 0;
        updateTextCondView();
    }

    var configuratorPills = document.querySelectorAll("#text-cond-configurator .config-pill");
    for (var pi = 0; pi < configuratorPills.length; pi++) {
        (function (pill) {
            pill.addEventListener("click", function () {
                var cat = pill.closest(".config-category").getAttribute("data-category");
                var val = pill.getAttribute("data-value");
                textCondSelection[cat] = val;
                resolveTextCondConfig(cat);
            });
        })(configuratorPills[pi]);
    }

    if (textCondPrevBtn) {
        textCondPrevBtn.addEventListener("click", function () {
            var totalPages = Math.ceil(getTextCondSamples(textCondCurrentConfig).length / 3);
            textCondPage = (textCondPage - 1 + totalPages) % totalPages;
            updateTextCondView();
        });
    }
    if (textCondNextBtn) {
        textCondNextBtn.addEventListener("click", function () {
            var totalPages = Math.ceil(getTextCondSamples(textCondCurrentConfig).length / 3);
            textCondPage = (textCondPage + 1) % totalPages;
            updateTextCondView();
        });
    }

    var projImageBox = document.getElementById("proj-image-box");
    if (projImageBox) {
        projImageBox.addEventListener("mouseenter", function () {
            projShowingGenerated = true;
            updateProjView();
        });
        projImageBox.addEventListener("mouseleave", function () {
            projShowingGenerated = false;
            updateProjView();
        });
    }

    updateView();
    updateProjView();
    resolveTextCondConfig();
})();

// Lazy-load videos: start loading when they scroll into view
(function () {
    var videos = document.querySelectorAll('video[preload="none"]');
    if (!('IntersectionObserver' in window)) {
        videos.forEach(function (v) { v.preload = 'auto'; v.load(); });
        return;
    }
    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                var video = entry.target;
                video.preload = 'auto';
                video.load();
                video.addEventListener('canplay', function () {
                    video.play().catch(function () {});
                }, { once: true });
                observer.unobserve(video);
            }
        });
    }, { rootMargin: '200px' });
    videos.forEach(function (v) { observer.observe(v); });
})();
