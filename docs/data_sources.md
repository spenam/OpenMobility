# Data sources and provenance

## Paris permanent road traffic counters

- Publisher: Ville de Paris, Direction de la Voirie et des Déplacements.
- Dataset identifier: `comptages-routiers-permanents`.
- Portal: https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/
- API: https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/records
- License shown on the Paris Open Data portal: ODbL.
- Coverage: rolling 13 months for the current dataset; historical records since 2010 are available in a separate archive dataset.
- Granularity: hourly.
- Measurements: traffic flow (vehicles/hour) and occupancy rate (percentage), with sensor/reference metadata available separately.
- Time zone: Europe/Paris.

The publisher cautions that the counters provide measurements at equipped road sections and do not alone characterize the full complexity of Paris traffic. Flow should be interpreted jointly with occupancy.

## Additional candidate sources (later phases)

- Multimodal counts (bicycles, scooters, motorized two-wheelers, cars, heavy vehicles, buses): https://opendata.paris.fr/explore/dataset/comptage-multimodal-comptages/
- Bicycle counter locations: https://opendata.paris.fr/explore/dataset/comptage-velo-compteurs/
- Historical bicycle counts: https://www.data.gouv.fr/datasets/comptage-velo-historique-donnees-compteurs
- Road-network geometry/reference: follow the geographic reference dataset linked from the permanent counters dataset.

Check the terms and attribution requirements of every source before redistributing data or derived datasets. Paris datasets commonly use ODbL; keep source provenance and license notices with derived outputs.
