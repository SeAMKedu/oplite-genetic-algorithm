[![DOI](https://zenodo.org/badge/719039919.svg)](https://zenodo.org/doi/10.5281/zenodo.10134568)

![logo](/images/OPLITE_logo_text.png)

# OPLITE - Geneettiinen algoritmi

## Geneettinen algoritmi

Geneettinen algoritmi on tietojenkäsittelytieteessä ratkaisujen hakuun käytettävä optimointimenetelmä. Se on evoluutioalgoritmi, jossa käytetään evoluutiobiologiasta tuttuja periytymisen, mutaatioiden ja rekombinaation prosesseja ratkaisujen hakemiseen. Geneettinen algoritmi esittää ratkaisuehdotukset kromosomeina.

Algoritmin alustusvaiheessa luodaan satunnaisesti valittu ratkaisuehdotusten joukko eli populaatio.

Populaation kullekin ratkaisuehdotukselle eli kromosomille lasketaan sopivuusarvo (engl. *fitness*), jonka perusteella populaatio jaetaan "hyviin" ja "huonoihin" ratkaisuihin. Ennalta määritellyillä ehdoilla huonot ratkaisut karsitaan pois, kun taas hyvät ratkaisut valitaan tuottamaan uusia ratkaisuehdotuksia.

Lisääntymisaskeleessa (engl. *crossover*) kutakin uutta ratkaisuehdotusta varten valitaan kaksi "vanhempaa", joiden "jälkeläinen" muodostetaan yhdistämällä satunnaisesti vanhempien ominaisuuksia.

Ratkaisuehdotuksia voidaan altistaa mutaatioille, jotka muuttavat ratkaisuehdotusten ominaisuuksia.

Algoritmia ajetaan kunnes löydetään riittävän hyvä ratkaisu tai sukupolvien määrälle asetettu yläraja tulee vastaan.

Lähde: [https://fi.wikipedia.org/wiki/Geneettinen_algoritmi](https://fi.wikipedia.org/wiki/Geneettinen_algoritmi).

Geneettisen algoritmin pseudokoodi on esitetty alla olevassa kuvassa.

![kuva](/images/pseudokoodi_geneettinen_algoritmi.png)

## Sovelluksen toiminta

Sovellus toimii yhdessä FMS-solun simulaatiomallin kanssa. Simulaatiomallissa on ennalta määritetty lista eurolavojen (EUR-lavojen) siirtoja, joiden järjestys optimoidaan geneettisen algoritmin avulla siten, että siirtojen kokonaiskesto on mahdollisimman lyhyt.

Simulaatiomalli (*FastemsFMS_GeneticAlgorithm.vcmx*) on tehty Visual Components -ohjelmalla ja malli löytyy *sims*-kansiosta. Huomaa, että ohjelman käyttö vaatii maksullisen lisenssin.

### Eurolavojen siirto ja hyllypaikat

Esimerkki eurolavan siirrosta hyllypaikalta P16 hyllypaikalle P01:
```
{'pid': 2, 'sku': 'epallet', 'src': 'P16', 'dst': 'P01'}
```
missä
* **pid** (*part ID*) on paletin yksilöllinen ID-numero
* **sku** (*stock keeping unit*) on varastointiyksikkö
* **src** (*source*) on hyllypaikka, jolta hyllystöhissi poimii lavan
* **dst** (*destination*) on hyllypaikka, jolle hyllystöhissi siirtää lavan

Alla olevassa kuvassa on esitetty FMS-solun varaston hyllypaikkojen numerointi simulaatiomallissa. Paikat P01-P07 ja P12-P27 on tarkoitettu eurolavoille (*epallet*), kun taas hyllypaikat P08-P11 on tarkoitettu koneistuspaleteille (*mpallet*).

![kuva](/images/vc_hyllypaikat.png)

### Siirtojen optimointi

Sovelluksessa voidaan käyttää siirtojen optimointiin joko [pygad](https://pygad.readthedocs.io/en/latest/)-pakettia tai itse tehtyä geneettisen algoritmin toteutusta. Valinta tehdään [config.yaml](/src/config.yaml)-tiedostossa asettamalla parametrille **use_pygad** arvo *True* tai *False*.

Simulaatiomallissa on mahdollista suorittaa siirtojen optimointi myös ilman ulkoista sovellusta. Hyllystöhissin ominaisuuksissa on boolean-tyypin parametri **UseExternalGA**, joka määrittää käytetäänkö optimointiin ulkoista sovellusta (*True*) vai ei (*False*). Parametrin arvoa voidaan muuttaa *Property*-ikkunassa (kts. alla oleva kuva).

![kuva](/images/vc_property.png)

Hyllystöhissin ominaisuuksissa on myös boolean-tyypin parametri **OptimizeTransfers**, jolla voidaan määrittää suorittaako hyllystöhissi siirrot alkuperäisessä (*False*) vai optimoidussa (*True*) järjestyksessä.

### Sopivuusarvon laskenta

Ratkaisuehdotuksen sopivuusarvon laskentaa varten simulaatiomallissa mitattiin aika, joka hyllystöhissiltä kuluu, kun se siirtää eurolavan miltä tahansa hyllypaikalta mille tahansa hyllypaikalle. Ajat on tallennettu [config.yaml](/src/config.yaml)-tiedostoon (*storage* -> *transfer_times*).

*sims*-kansiossa on myös simulaatiomalli *FastemsFMS_GA_SimulateFitness.vcmx*, jossa lavojen siirtoaikoja ei ole ennalta määritetty, vaan ne mitataan simulaation ajon aikana. Tässä mallissa optimointi tapahtuu simulaatio-ohjelman sisällä, eikä siinä ole yhteyttä ulkopuoliseen optimointisovellukseen.

### Sovelluksen käyttö

Käytettäessä ulkoista sovellusta, sen tulee olla käynnissä ennen kuin simulaatio aloitetaan. Kun simulaatio aloitetaan, simulaatiomalli lähettää siirrot ulkoiselle sovellukselle, joka optimoi siirtojen järjestyksen ja lähettää sitten optimoidut siirrot takaisin simulaatiomallille, jossa hyllystöhissi suorittaa siirrot geneettisen algoritmin optimoimassa järjestyksessä.

Simulaatiomalli tulostaa siirtoihin kuluneen ajan *Output*-ikkunaan.

## Tekijätiedot

Hannu Hakalahti, Asiantuntija TKI, Seinäjoen ammattikorkeakoulu

## Hanketiedot

* Hankkeen nimi: OPLITE - Optimaalista lisäarvoa teknologiasta
* Rahoittaja: Etelä-Pohjanmaan liitto
* Aikataulu: 1.1.2023 - 31.12.2025
* Hankkeen kotisivut: [https://projektit.seamk.fi/alykkaat-teknologiat/oplite/](https://projektit.seamk.fi/alykkaat-teknologiat/oplite/)
---
![EU_logo](/images/Euroopan_unionin_osarahoittama_POS.png)
![EPLiitto](/images/EPLiitto_logo_vaaka_vari.jpg)
![SeAMK](/images/SEAMK_vaaka_fi_en_RGB_1200x486.jpg)
