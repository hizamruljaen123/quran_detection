/**
 * TajweedDetector.js
 * 
 * Library untuk deteksi otomatis kaidah tajweed dalam teks Arab
 * dan memberikan pewarnaan sesuai dengan aturan tajweed.
 * 
 * Cara Penggunaan:
 * 1. Import library ini
 * 2. Inisialisasi: const tajweed = new TajweedDetector();
 * 3. Gunakan: tajweed.parse(text);
 * 
 * @author Quran Detection Project
 * @version 1.0.0
 */

class TajweedDetector {
    /**
     * Membuat instance baru TajweedDetector
     * @param {Object} options - Opsi konfigurasi detector
     */
    constructor(options = {}) {
        // Default options
        this.options = {
            showTooltips: true,
            tooltipLanguage: 'id',
            showLegend: true,
            ...options
        };

        // Tajweed rules definitions dengan regex patterns
        this.rules = {
            // Ghunnah - Nun atau Meem dengan shadda
            ghunnah: {
                pattern: /[نم]ّ/g,
                color: '#ff6b6b',
                description: {
                    id: 'Ghunnah - Bunyi dengung pada Nun atau Meem bertashdid',
                    en: 'Ghunnah - Nasalization of Nun or Meem with shadda'
                }
            },
            
            // Qalqalah - Huruf qalqalah (ق ط ب ج د)
            qalqalah: {
                pattern: /[قطبجد](?=\s|$|[^\u064B-\u065F])/g,
                color: '#4ecdc4',
                description: {
                    id: 'Qalqalah - Memantul pada huruf ق ط ب ج د',
                    en: 'Qalqalah - Bouncing sound on letters ق ط ب ج د'
                }
            },
            
            // Ikhfa - Nun sakinah/tanwin sebelum huruf ikhfa
            ikhfa: {
                pattern: /ن(?=[\u064B-\u065F]*[تثجدذزسشصضطظفقك])|[ً ٍ ٌ](?=[تثجدذزسشصضطظفقك])/g,
                color: '#45b7d1',
                description: {
                    id: 'Ikhfa - Menyamarkan bunyi Nun sukun/tanwin',
                    en: 'Ikhfa - Hiding sound of Nun sukun/tanwin'
                }
            },
            
            // Idgham - Nun sakinah/tanwin sebelum huruf idgham (ي ر ل و م ن)
            idgham: {
                pattern: /ن(?=[\u064B-\u065F]*[يرلومن])|[ً ٍ ٌ](?=[يرلومن])/g,
                color: '#96ceb4',
                description: {
                    id: 'Idgham - Memasukkan bunyi Nun sukun ke huruf berikutnya',
                    en: 'Idgham - Merging sound of Nun sukun into the next letter'
                }
            },
            
            // Idgham Ghunnah - Nun sakinah/tanwin sebelum huruf (ي و م ن)
            idghamGhunnah: {
                pattern: /ن(?=[\u064B-\u065F]*[يومن])|[ً ٍ ٌ](?=[يومن])/g,
                color: '#96ceb4',
                description: {
                    id: 'Idgham Ghunnah - Memasukkan dengan dengung',
                    en: 'Idgham Ghunnah - Merging with nasalization'
                }
            },
            
            // Idgham Bila Ghunnah - Nun sakinah/tanwin sebelum huruf (ل ر)
            idghamBilaGhunnah: {
                pattern: /ن(?=[\u064B-\u065F]*[لر])|[ً ٍ ٌ](?=[لر])/g,
                color: '#20c997',
                description: {
                    id: 'Idgham Bila Ghunnah - Memasukkan tanpa dengung',
                    en: 'Idgham Bila Ghunnah - Merging without nasalization'
                }
            },
            
            // Iqlab - Nun sakinah/tanwin sebelum Ba (ب)
            iqlab: {
                pattern: /ن(?=[\u064B-\u065F]*ب)|[ً ٍ ٌ](?=ب)/g,
                color: '#feca57',
                description: {
                    id: 'Iqlab - Mengubah Nun sukun/tanwin menjadi Mim',
                    en: 'Iqlab - Changing Nun sukun/tanwin into Mim'
                }
            },
            
            // Izhar - Nun sakinah/tanwin sebelum huruf izhar (ء ه ع ح غ خ)
            izhar: {
                pattern: /ن(?=[\u064B-\u065F]*[ءهعحغخ])|[ً ٍ ٌ](?=[ءهعحغخ])/g,
                color: '#ff9ff3',
                description: {
                    id: 'Izhar - Membaca jelas Nun sukun/tanwin',
                    en: 'Izhar - Clear pronunciation of Nun sukun/tanwin'
                }
            },
            
            // Madd - Tanda pemanjangan bacaan
            madd: {
                pattern: /[اويى](?=[\u064B-\u065F]*[اويى])|آ|[اويى]ّ/g,
                color: '#a29bfe',
                description: {
                    id: 'Madd - Memanjangkan bacaan',
                    en: 'Madd - Elongation of sound'
                }
            },
            
            // Lam Shamsiyyah
            lamShamsiyyah: {
                pattern: /ال(?=[تثدذرزسشصضطظلن])/g,
                color: '#54a0ff',
                description: {
                    id: 'Lam Shamsiyyah - Lam yang tidak dibaca',
                    en: 'Lam Shamsiyyah - Silent Lam'
                }
            },
            
            // Lam Qamariyyah
            lamQamariyyah: {
                pattern: /ال(?=[ابجحخعغفقكمهويء])/g,
                color: '#5f27cd',
                description: {
                    id: 'Lam Qamariyyah - Lam yang dibaca jelas',
                    en: 'Lam Qamariyyah - Clear pronunciation of Lam'
                }
            },
            
            // Ra Tafkhim (tebal)
            raTafkhim: {
                pattern: /ر(?=[\u064E\u064F])|(?:[\u064E\u064F])ر/g,
                color: '#ff6348',
                description: {
                    id: 'Ra Tafkhim - Ra tebal',
                    en: 'Ra Tafkhim - Heavy Ra'
                }
            },
            
            // Ra Tarqiq (tipis)
            raTarqiq: {
                pattern: /ر(?=[\u0650])|(?:[\u0650])ر/g,
                color: '#2ed573',
                description: {
                    id: 'Ra Tarqiq - Ra tipis',
                    en: 'Ra Tarqiq - Light Ra'
                }
            },
            
            // Waqf - Tanda berhenti
            waqf: {
                pattern: /[ۖۗۘۙۚۛۜۢۤۧۨ۩]/g,
                color: '#a4b0be',
                description: {
                    id: 'Tanda Waqf - Tanda berhenti',
                    en: 'Waqf Signs - Stopping signs'
                }
            },
            
            // Tanda Saktah
            saktah: {
                pattern: /ۜ/g, 
                color: '#778ca3',
                description: {
                    id: 'Saktah - Berhenti sejenak tanpa bernafas',
                    en: 'Saktah - Brief pause without taking breath'
                }
            },
            
            // Mim Sukun
            mimSukun: {
                pattern: /مْ(?=[^ب])|مْ$/g,
                color: '#f368e0',
                description: {
                    id: 'Mim Sukun - Pengucapan jelas mim sukun',
                    en: 'Mim Sukun - Clear pronunciation of mim sukun'
                }
            },
            
            // Ikhfa Syafawi - Mim sukun bertemu dengan Ba
            ikhfaSyafawi: {
                pattern: /مْ(?=ب)/g,
                color: '#ff9f43',
                description: {
                    id: 'Ikhfa Syafawi - Mim sukun bertemu Ba',
                    en: 'Ikhfa Syafawi - Mim sukun followed by Ba'
                }
            },
            
            // Idgham Mimi - Mim sukun bertemu dengan Mim
            idghamMimi: {
                pattern: /مْ(?=م)/g,
                color: '#c8d6e5',
                description: {
                    id: 'Idgham Mimi - Mim sukun bertemu Mim',
                    en: 'Idgham Mimi - Mim sukun followed by Mim'
                }
            }
        };
        
        // Menyimpan semua data tentang tajweed
        this.tajweedData = {};
    }

    /**
     * Mengurai teks Arab dan menerapkan aturan tajweed
     * @param {String} text - Teks Arab yang akan diurai
     * @param {HTMLElement} targetElement - Elemen untuk menampilkan teks
     * @return {String} - HTML dengan format tajweed
     */
    parse(text, targetElement = null) {
        let processedText = text;
        this.tajweedData = {};
        
        // Terapkan setiap aturan
        Object.entries(this.rules).forEach(([ruleName, ruleData]) => {
            let matches = [];
            let match;
            
            // Temukan semua kecocokan untuk aturan ini
            while ((match = ruleData.pattern.exec(processedText)) !== null) {
                matches.push({
                    index: match.index,
                    length: match[0].length,
                    text: match[0],
                    rule: ruleName
                });
            }
            
            // Reset lastIndex untuk regex untuk penggunaan berikutnya
            ruleData.pattern.lastIndex = 0;
            
            // Simpan semua kecocokan untuk aturan ini
            this.tajweedData[ruleName] = matches;
        });
        
        // Gabungkan semua kecocokan dan urutkan berdasarkan posisi
        const allMatches = [];
        Object.entries(this.tajweedData).forEach(([ruleName, matches]) => {
            matches.forEach(match => {
                match.rule = ruleName;
                allMatches.push(match);
            });
        });
        
        // Urutkan dari belakang ke depan untuk menghindari pergeseran indeks
        allMatches.sort((a, b) => b.index - a.index);
        
        // Terapkan span dengan style untuk setiap kecocokan
        allMatches.forEach(match => {
            const rule = this.rules[match.rule];
            const lang = this.options.tooltipLanguage;
            const description = rule.description[lang] || rule.description.en;
            
            const before = processedText.substring(0, match.index);
            const after = processedText.substring(match.index + match.length);
            
            let replacement;
            if (this.options.showTooltips) {
                replacement = `<span class="tajweed-${match.rule}" style="color: ${rule.color};" title="${description}">${match.text}</span>`;
            } else {
                replacement = `<span class="tajweed-${match.rule}" style="color: ${rule.color};">${match.text}</span>`;
            }
            
            processedText = before + replacement + after;
        });
        
        // Jika elemen target diberikan, perbarui kontennya
        if (targetElement) {
            targetElement.innerHTML = processedText;
            this._enableTooltips(targetElement);
        }
        
        return processedText;
    }

    /**
     * Mengaktifkan tooltips pada elemen yang telah diformat
     * @param {HTMLElement} element - Elemen yang berisi teks yang diformat
     * @private
     */
    _enableTooltips(element) {
        if (!this.options.showTooltips) return;
        
        const tajweedSpans = element.querySelectorAll('[class^="tajweed-"]');
        
        tajweedSpans.forEach(span => {
            span.addEventListener('mouseover', (e) => {
                const tooltip = document.createElement('div');
                tooltip.className = 'tajweed-tooltip';
                tooltip.textContent = span.getAttribute('title');
                tooltip.style.cssText = `
                    position: absolute;
                    background: rgba(0,0,0,0.9);
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 0.85rem;
                    z-index: 1000;
                    pointer-events: none;
                    white-space: nowrap;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                    font-family: Arial, sans-serif;
                `;
                
                document.body.appendChild(tooltip);
                
                const rect = span.getBoundingClientRect();
                tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + window.scrollX + 'px';
                tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + window.scrollY + 'px';
            });
            
            span.addEventListener('mouseout', () => {
                const tooltip = document.querySelector('.tajweed-tooltip');
                if (tooltip && tooltip.parentNode) {
                    tooltip.parentNode.removeChild(tooltip);
                }
            });
        });
    }

    /**
     * Membuat legenda untuk aturan tajweed
     * @param {HTMLElement} container - Elemen container untuk legenda
     * @return {HTMLElement} - Elemen legenda
     */
    createLegend(container) {
        const legend = document.createElement('div');
        legend.className = 'tajweed-legend';
        legend.style.cssText = `
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        `;
        
        const title = document.createElement('h6');
        title.innerHTML = '<i class="fas fa-info-circle me-2"></i>Keterangan Warna Tajweed:';
        title.style.cssText = `
            margin-bottom: 15px;
            color: #2c3e50;
            font-weight: bold;
        `;
        
        legend.appendChild(title);
        
        const rulesContainer = document.createElement('div');
        rulesContainer.style.cssText = `
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        `;
        
        // Isi legenda dengan semua aturan
        Object.entries(this.rules).forEach(([ruleName, ruleData]) => {
            const lang = this.options.tooltipLanguage;
            const description = ruleData.description[lang] || ruleData.description.en;
            
            // Split name dan deskripsi
            const nameParts = description.split('-');
            const ruleTitle = nameParts[0].trim();
            
            const ruleSpan = document.createElement('span');
            ruleSpan.className = `tajweed-rule tajweed-${ruleName}`;
            ruleSpan.textContent = ruleTitle;
            ruleSpan.style.cssText = `
                display: inline-block;
                padding: 5px 10px;
                margin: 3px;
                border-radius: 15px;
                font-size: 0.85rem;
                color: white;
                font-weight: 500;
                background-color: ${ruleData.color};
            `;
            
            ruleSpan.setAttribute('title', description);
            rulesContainer.appendChild(ruleSpan);
        });
        
        legend.appendChild(rulesContainer);
        
        if (container) {
            container.appendChild(legend);
        }
        
        return legend;
    }

    /**
     * Toggle tampilan tajweed pada elemen
     * @param {HTMLElement} element - Elemen yang berisi teks
     * @param {String} originalText - Teks asli tanpa format tajweed
     * @param {Boolean} showTajweed - Apakah menampilkan tajweed atau tidak
     */
    toggleTajweed(element, originalText, showTajweed) {
        if (showTajweed) {
            element.innerHTML = this.parse(originalText);
            this._enableTooltips(element);
        } else {
            element.textContent = originalText;
        }
    }
    
    /**
     * Mendapatkan data statistik tentang tajweed dalam teks
     * @param {String} text - Teks yang akan dianalisis
     * @return {Object} - Statistik tajweed
     */
    getStatistics(text) {
        // Parse teks terlebih dahulu (tanpa mempengaruhi DOM)
        this.parse(text);
        
        const stats = {
            totalRules: 0,
            rulesCounts: {}
        };
        
        // Hitung semua aturan tajweed
        Object.entries(this.tajweedData).forEach(([ruleName, matches]) => {
            stats.rulesCounts[ruleName] = matches.length;
            stats.totalRules += matches.length;
        });
        
        return stats;
    }
}

// Menambahkan metode statis untuk memudahkan inisialisasi
TajweedDetector.init = function(options = {}) {
    return new TajweedDetector(options);
};

// Eksport untuk penggunaan ES modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TajweedDetector;
}
